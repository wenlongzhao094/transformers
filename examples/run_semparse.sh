#!/bin/bash
#
#SBATCH --job-name=semparse_bert
#SBATCH --partition=2080ti-long
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4

DATA_DIR=/mnt/nfs/scratch1/wenlongzhao/CS696DS_Data/snips
TASK_NAME=snips
MODEL_TYPE=bert
MODEL_PATH=bert-base-cased
EVAL_MODE=valid
OUTPUT_DIR=/mnt/nfs/scratch1/wenlongzhao/CS696DS_Model/xxx

python ../run_semparse.py \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL_PATH \
  --task_name $TASK_NAME \
  --do_train \
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
