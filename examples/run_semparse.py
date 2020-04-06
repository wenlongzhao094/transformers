# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for semantic parsing, i.e. sequence (intent) and token (slot) classification,
on AITS and SNIPS (Bert, Albert)."""


import argparse
import glob
import json
import logging
import os
import random
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from dataclasses import dataclass

from seqeval.metrics import f1_score, precision_score, recall_score

from transformers import (
    MODEL_FOR_SEQUENCE_AND_TOKEN_CLASSIFICATION_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForSequenceAndTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_SEQUENCE_AND_TOKEN_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in MODEL_CONFIG_CLASSES), (),)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


@dataclass(frozen=True)
class InputExample:
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        text_a: The untokenized text of the query.
        label_s: slot labels.
        label_i: intent labels.
    """

    guid: str
    text_a: str
    label_s: str
    label_i: str

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors
        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_slot_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def get_intent_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class AtisProcessor(DataProcessor):
    """Processor for the ATIS data set (standard version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label_slot"].numpy()),
            str(tensor_dict["label_intent"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "test")

    def get_slot_labels(self):
        """See base class."""
        return ['B-stoploc.city_name', 'I-arrive_time.time_relative', 'B-depart_time.period_mod', 'I-toloc.city_name',
                'B-depart_time.time_relative', 'B-return_time.period_of_day', 'I-arrive_date.day_number',
                'B-arrive_date.day_name', 'I-toloc.airport_name', 'I-class_type', 'B-flight', 'B-or',
                'I-fromloc.airport_name', 'B-meal', 'B-toloc.country_name', 'I-fare_amount', 'B-depart_date.day_number',
                'B-toloc.state_code', 'I-meal_code', 'I-transport_type', 'I-stoploc.city_name', 'B-flight_mod',
                'B-fromloc.airport_name', 'B-stoploc.airport_name', 'I-flight_stop', 'B-depart_date.day_name',
                'B-arrive_time.time_relative', 'B-depart_date.month_name', 'B-arrive_date.today_relative',
                'B-return_date.today_relative', 'B-depart_date.year', 'B-depart_time.time', 'B-days_code', 'B-economy',
                'I-time', 'I-depart_time.period_of_day', 'I-arrive_time.time', 'B-fromloc.state_name',
                'B-stoploc.airport_code', 'I-flight_mod', 'B-airport_name', 'I-meal_description',
                'B-toloc.airport_code', 'B-flight_days', 'I-fromloc.state_name', 'B-depart_date.today_relative',
                'B-state_code', 'I-airline_name', 'B-class_type', 'B-arrive_date.date_relative', 'I-fare_basis_code',
                'B-arrive_time.period_of_day', 'B-toloc.airport_name', 'I-state_name', 'B-flight_stop',
                'B-return_date.date_relative', 'B-arrive_time.start_time', 'I-economy', 'B-compartment',
                'B-airline_name', 'B-return_date.day_number', 'B-transport_type', 'I-arrive_time.start_time',
                'I-arrive_time.end_time', 'B-booking_class', 'B-toloc.state_name', 'I-airport_name', 'I-city_name',
                'I-depart_time.start_time', 'I-depart_time.end_time', 'B-arrive_time.time', 'B-depart_time.end_time',
                'O', 'B-airport_code', 'B-arrive_time.period_mod', 'I-fromloc.city_name', 'B-today_relative',
                'B-state_name', 'I-flight_time', 'I-depart_time.time_relative', 'B-return_date.day_name',
                'B-depart_time.start_time', 'B-meal_description', 'B-meal_code', 'B-mod', 'I-flight_number',
                'B-toloc.city_name', 'B-day_name', 'B-aircraft_code', 'B-day_number', 'B-time',
                'B-fromloc.airport_code', 'B-fromloc.state_code', 'B-city_name', 'B-flight_time', 'B-flight_number',
                'I-return_date.today_relative', 'B-arrive_date.month_name', 'I-depart_time.time',
                'B-depart_date.date_relative', 'B-period_of_day', 'B-return_date.month_name',
                'I-return_date.date_relative', 'B-connect', 'B-month_name', 'I-round_trip', 'I-return_date.day_number',
                'B-return_time.period_mod', 'B-round_trip', 'I-restriction_code', 'B-fare_basis_code',
                'I-today_relative', 'I-cost_relative', 'B-stoploc.state_code', 'I-depart_date.day_number',
                'B-arrive_time.end_time', 'B-arrive_date.day_number', 'B-cost_relative', 'B-restriction_code',
                'B-fare_amount', 'I-toloc.state_name', 'B-time_relative', 'I-arrive_time.period_of_day',
                'B-airline_code', 'B-fromloc.city_name', 'I-depart_date.today_relative', 'B-depart_time.period_of_day']

    def get_intent_labels(self):
        """See base class."""
        return ['atis_city', 'atis_ground_service#atis_ground_fare', 'atis_flight#atis_airfare', 'atis_abbreviation',
                'atis_meal', 'atis_airline#atis_flight_no', 'atis_ground_service', 'atis_ground_fare', 'atis_distance',
                'atis_airfare#atis_flight_time', 'atis_flight', 'atis_flight_no', 'atis_quantity',
                'atis_airfare#atis_flight', 'atis_restriction', 'atis_airport', 'atis_airfare', 'atis_day_name',
                'atis_flight_time', 'atis_capacity', 'atis_cheapest', 'atis_aircraft', 'atis_airline',
                'atis_flight_no#atis_airline', 'atis_aircraft#atis_flight#atis_flight_no', 'atis_flight#atis_airline']

    def _create_examples(self, data_dir, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        input_file_data = os.path.join(data_dir, "seq.in")
        input_file_slots = os.path.join(data_dir, "seq.out")
        input_file_intents = os.path.join(data_dir, "label")
        with open(input_file_data, "r", encoding="utf-8-sig") as f1, \
                open(input_file_slots, "r", encoding="utf-8-sig") as f2, \
                open(input_file_intents, "r", encoding="utf-8-sig") as f3:
            for i, (x, y, z) in enumerate(zip(f1, f2, f3)):
                guid = "%s-%s" % (set_type, i)
                text_a = x.strip()
                label_s = y.strip()
                label_i = z.strip()
                examples.append(InputExample(guid=guid, text_a=text_a, label_s=label_s, label_i=label_i))
        return examples


class SnipsProcessor(DataProcessor):
    """Processor for the SNIPS data set (standard version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label_slot"].numpy()),
            str(tensor_dict["label_intent"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "test")

    def get_slot_labels(self):
        """See base class."""
        return ['B-object_name', 'B-best_rating', 'B-condition_description', 'I-location_name', 'I-served_dish',
                'B-restaurant_name', 'B-country', 'I-movie_name', 'B-party_size_description', 'I-restaurant_type',
                'I-state', 'I-cuisine', 'I-object_type', 'B-artist', 'I-geographic_poi', 'B-object_part_of_series_type',
                'I-playlist', 'O', 'B-track', 'B-party_size_number', 'B-facility', 'B-year', 'I-album', 'B-genre',
                'B-sort', 'I-entity_name', 'B-movie_type', 'B-entity_name', 'I-party_size_description', 'B-album',
                'I-artist', 'I-city', 'B-object_location_type', 'I-object_part_of_series_type', 'I-spatial_relation',
                'I-object_name', 'B-location_name', 'B-cuisine', 'I-poi', 'I-music_item', 'I-object_select',
                'B-playlist_owner', 'B-served_dish', 'B-playlist', 'I-restaurant_name', 'B-spatial_relation',
                'B-service', 'I-genre', 'I-facility', 'B-geographic_poi', 'B-movie_name', 'B-poi', 'B-current_location',
                'B-music_item', 'B-city', 'I-playlist_owner', 'B-object_type', 'I-track', 'I-current_location',
                'B-object_select', 'B-state', 'I-timeRange', 'B-rating_unit', 'B-rating_value', 'B-restaurant_type',
                'I-sort', 'B-condition_temperature', 'I-service', 'I-movie_type', 'B-timeRange',
                'I-object_location_type', 'I-country']

    def get_intent_labels(self):
        """See base class."""
        return ['GetWeather', 'PlayMusic', 'BookRestaurant', 'RateBook', 'AddToPlaylist', 'SearchCreativeWork',
                'SearchScreeningEvent']

    def _create_examples(self, data_dir, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        input_file_data = os.path.join(data_dir, "seq.in")
        input_file_slots = os.path.join(data_dir, "seq.out")
        input_file_intents = os.path.join(data_dir, "label")
        with open(input_file_data, "r", encoding="utf-8-sig") as f1, \
                open(input_file_slots, "r", encoding="utf-8-sig") as f2, \
                open(input_file_intents, "r", encoding="utf-8-sig") as f3:
            for i, (x, y, z) in enumerate(zip(f1, f2, f3)):
                guid = "%s-%s" % (set_type, i)
                text_a = x.strip()
                label_s = y.strip()
                label_i = z.strip()
                assert len(text_a.split()) == len(label_s.split())
                examples.append(InputExample(guid=guid, text_a=text_a, label_s=label_s, label_i=label_i))
        return examples


class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label_slots=None, label_intents=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_slots = label_slots
        self.label_intents = label_intents

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def semparse_convert_examples_to_features(
    examples,
    tokenizer,
    label_slot_list,
    label_intent_list,
    max_seq_length=512,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    label_slot_map = {label: i for i, label in enumerate(label_slot_list)}
    label_intent_map = {label: i for i, label in enumerate(label_intent_list)}

    features = []

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_slot_ids = []
        for word, label in zip(example.text_a.split(), example.label_s.split()):
            word_tokens = tokenizer.tokenize(word)

            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_slot_ids.extend([label_slot_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = tokenizer.num_added_tokens()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_slot_ids = label_slot_ids[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        label_slot_ids += [pad_token_label_id]

        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_slot_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_slot_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_slot_ids = [pad_token_label_id] + label_slot_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_slot_ids = ([pad_token_label_id] * padding_length) + label_slot_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_slot_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_slot_ids) == max_seq_length

        label_intent_id = label_intent_map[example.label_i]

        features.append(
            InputFeatures(input_ids=input_ids, attention_mask=input_mask, \
                          token_type_ids=segment_ids, label_slots=label_slot_ids, label_intents=label_intent_id)
        )

    return features


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare task
    args.task_name = args.task_name.lower()
    processors = {"atis": AtisProcessor, "snips": SnipsProcessor}
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )


    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
