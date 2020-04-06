#!/bin/bash
#
#SBATCH --job-name=sst_2_bert_cc
#SBATCH --partition=m40-long
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4

GLUE_DIR=../glue_data
TASK_NAME=SST-2
MODEL_TYPE=bert
MODEL_PATH=/mnt/nfs/scratch1/prafullpraka/696DS/BERT/GLUE_benchmark/Ckpt_results_BERT_cased/SST-2
SETTING=bert-base-cased
CCEMD_PATH=/mnt/nfs/scratch1/ssaurabhkuma/2020/696DS/BERT/GLUE_benchmark/Ckpt_results_BERT_cased/src/neuralcompressor/models/M32_K16/SST-2
CODEBOOK_NUM=32
CODEBOOK_SIZE=16
OUTPUT_DIR=/mnt/nfs/scratch1/wenlongzhao/Result_Efficiency/xxx

python ../run_glue_cc.py \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL_PATH \
  --compositional_code_embedding_path $CCEMD_PATH \
  --codebook_num $CODEBOOK_NUM \
  --codebook_size $CODEBOOK_SIZE \
  --task_name $TASK_NAME \
  --config_name $SETTING\
  --tokenizer_name $SETTING\
  --do_train \
  --train_codebook \
  --not_train_transformer \
  --do_eval \
  --eval_all_checkpoints \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 8\
  --gradient_accumulation_steps 1\
  --learning_rate 2e-5 \
  --num_train_epochs 5.0 \
  --output_dir $OUTPUT_DIR/$TASK_NAME


