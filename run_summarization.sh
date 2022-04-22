MODEL_PATHS=("/storage/private/student-lkm/t5_sl_small-pytorch" "/storage/private/student-lkm/t5_sl_large-pytorch/")
MODEL_SHORTHANDS=("small" "large")
TRAIN_FILE="/home/katjal/QAslovene/datasets/encoded/BoolQ/train.csv"
VALIDATION_FILE="/home/katjal/QAslovene/datasets/encoded/BoolQ/val.csv"
TEST_FILE="/home/katjal/QAslovene/datasets/encoded/BoolQ/test_answered.csv"
MAX_SOURCE_LENGTH=512
MAX_TARGET_LENGTH=100
PER_DEVICE_TRAIN_BATCH_SIZE=4
PER_DEVICE_EVAL_BATCH_SIZE=4
NUM_TRAIN_EPOCHS=10
GRADIENT_ACCUMULATION=8

for i in 0 0
do
OUTPUT_DIR="/home/katjal/QAslovene/models/BoolQ-${MODEL_SHORTHANDS[$i]}"
MODEL_NAME_OR_PATH=${MODEL_PATHS[$i]}
CUDA_VISIBLE_DEVICES=0 python run_summarization.py\
 --model_name_or_path $MODEL_NAME_OR_PATH \
 --no_use_fast_tokenizer \
 --train_file $TRAIN_FILE \
 --validation_file $VALIDATION_FILE \
 --test_file $TEST_FILE \
 --text_column "input" \
 --summary_column "output" \
 --max_source_length $MAX_SOURCE_LENGTH \
 --max_target_length $MAX_TARGET_LENGTH \
 --output_dir $OUTPUT_DIR \
 --do_train --do_eval --do_predict \
 --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
 --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
 --num_train_epochs $NUM_TRAIN_EPOCHS \
 --save_strategy epoch \
 --evaluation_strategy epoch \
 --seed 42 \
 --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
 --predict_with_generate\
 --load_best_model_at_end \
 --metric_for_best_model "eval_rougeL" \
 --greater_is_better=True \
 --save_total_limit=1

done

