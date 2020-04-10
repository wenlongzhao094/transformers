#!/bin/bash
#
#SBATCH --job-name=sst_2_bert
#SBATCH --partition=2080ti-long
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4

GLUE_DIR=../glue_data
TASK_NAME=SST-2
MODEL_TYPE=bert
MODEL_PATH=bert-base-cased
OUTPUT_DIR=/mnt/nfs/scratch1/wenlongzhao/CS696DS_Model/xxx

python ../run_glue.py \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL_PATH \
  --task_name $TASK_NAME \
  --do_train \
  --save_steps 0 \
  --early_stopping 3 \
#  --do_eval \
#  --eval_all_checkpoints \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 8\
  --gradient_accumulation_steps 1\
  --learning_rate 2e-5 \
  --num_train_epochs 10.0 \
  --output_dir $OUTPUT_DIR/$TASK_NAME
