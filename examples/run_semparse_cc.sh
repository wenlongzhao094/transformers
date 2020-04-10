#!/bin/bash
#
#SBATCH --job-name=semparse_bert_cc
#SBATCH --partition=2080ti-long
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4

DATA_DIR=/mnt/nfs/scratch1/wenlongzhao/CS696DS_Data/snips
TASK_NAME=snips
MODEL_TYPE=bert
MODEL_PATH=/mnt/nfs/scratch1/wenlongzhao/CS696DS_Model/xxx
SETTING=bert-base-cased
CCEMD_PATH=/mnt/nfs/scratch1/ssaurabhkuma/2020/696DS/cc_embeddings/neuralcompressor/models/M8K8/xxx
CODEBOOK_NUM=32
CODEBOOK_SIZE=16
EVAL_MODE=valid
OUTPUT_DIR=/mnt/nfs/scratch1/wenlongzhao/CS696DS_Model/xxx

python ../run_semparse_cc.py \
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
  --save_steps 0 \
  --early_stopping 3 \
#  --do_eval $EVAL_MODE\
#  --eval_all_checkpoints \
  --data_dir $DATA_DIR \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --gradient_accumulation_steps 1\
  --learning_rate 5e-5 \
  --warmup_portion 0.1 \
  --weight_decay 0.01 \
  --num_train_epochs 40.0 \
  --output_dir $OUTPUT_DIR/$TASK_NAME


  /mnt/nfs/scratch1/prafullpraka/696DS/BERT/GLUE_benchmark/Ckpt_results_BERT_cased/SST-2/
  /mnt/nfs/scratch1/ssaurabhkuma/2020/696DS/BERT/GLUE_benchmark/Ckpt_results_BERT_cased/src/neuralcompressor/models/M32_K16/SST-2
