MODEL_PATHS=("/storage/private/student-lkm/t5_sl_small-pytorch" "/storage/private/student-lkm/t5_sl_large-pytorch/")
MODEL_SHORTHANDS=("small" "large")
MAX_SOURCE_LENGTH=512
MAX_TARGET_LENGTH=100
PER_DEVICE_TRAIN_BATCH_SIZE=8
PER_DEVICE_EVAL_BATCH_SIZE=8
NUM_TRAIN_EPOCHS=25
GRADIENT_ACCUMULATION=1

for i in 0 0
do
OUTPUT_DIR="/home/katjal/QAslovene/models/unified-without-noanswer-based"
#MODEL_NAME_OR_PATH=${MODEL_PATHS[$i]}
#MODEL_NAME_OR_PATH="google/mt5-small"
MODEL_NAME_OR_PATH="models/unified-without-noanswer-all/checkpoint-259279"
CUDA_VISIBLE_DEVICES=0 python run_summarization.py\
 --model_name_or_path $MODEL_NAME_OR_PATH \
 --no_use_fast_tokenizer \
 --train_file "train.csv" \
 --input_column "input" \
 --output_column "output" \
 --max_source_length $MAX_SOURCE_LENGTH \
 --max_target_length $MAX_TARGET_LENGTH \
 --output_dir $OUTPUT_DIR \
 --do_train \
 --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
 --num_train_epochs $NUM_TRAIN_EPOCHS \
 --save_strategy epoch \
 --seed 42 \
 --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
 --datasets "SQUAD2-project,BoolQ,COPA,MCTest,MultiRC" \
 --datasets_path "/home/katjal/QAslovene/datasets/encoded/" \
 --num_beams=4

done

# --validation_file "val.csv" \
# --do_eval
# --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
#--evaluation_strategy epoch \
#--predict_with_generate\
# --load_best_model_at_end \
# --metric_for_best_model "eval_rougeL" \
# --greater_is_better=True \
#--save_total_limit=1 \
#  --filter_no_answer=True
#--datasets "SQUAD2-project,BoolQ,COPA,MCTest,MultiRC" \
#--datasets "SQUAD2-project,BoolQ,COPA,MCTest,MultiRC,SQUAD2-eng,BoolQ-eng,COPA-eng,MCTest-eng,MultiRC-eng" \
#--lowercase=True
