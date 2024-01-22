#!/bin/bash
#SBATCH --job-name=convosense_train
#SBATCH --output=convosense_train
#SBATCH --gres=gpu:1

export PYTHONUNBUFFERED=1
export STDOUT_LINE_BUFFERED=1

cd ~/ConvoSense/train
TRAIN_DATA="~/ConvoSense/data/convosense/train.jsonl"

OUTPUT_FOLDER="experiments"
prefix="t53b_convosense"
lr=5e-6
wd=5e-3
bs=4
gas=2
efbs=$((bs * gas))
SAVE_NAME="${prefix}_lr=${lr}_wd=${wd}_bs=${efbs}"

python run_seq2seq.py \
--learning_rate $lr \ 
--adafactor \
--num_train_epochs 5 \
--train_file $TRAIN_DATA \
--text_column "input" \
--summary_column "output" \
--source_prefix="provide a reasonable answer to the question based on the dialogue:\n" \
--output_dir=$OUTPUT_FOLDER"/${SAVE_NAME}" \
--model_name_or_path='t5-3b' \
--seed 1234 \
--max_source_length 768 \
--resize_position_embeddings True \
--per_device_train_batch_size $bs \
--gradient_accumulation_steps $gas \
--weight_decay $wd \
--bf16 True \
--do_train True \
--do_eval False \
--logging_strategy "steps" \
--save_strategy "steps" \
--logging_steps 100 \
--save_steps 0.20 \
--report_to "none" \
--run_name SAVE_NAME \
--overwrite_output_dir