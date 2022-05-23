MODEL_PATHS=("/storage/private/student-lkm/t5_sl_small-pytorch" "/storage/private/student-lkm/t5_sl_large-pytorch/")
MODEL_SHORTHANDS=("small" "large")
MODELS=("BoolQ" "COPA" "MCTest" "MultiRC" "SQUAD2" "unified")
MAX_SOURCE_LENGTH=512
MAX_TARGET_LENGTH=100
PER_DEVICE_TRAIN_BATCH_SIZE=8
PER_DEVICE_EVAL_BATCH_SIZE=8
NUM_TRAIN_EPOCHS=100
GRADIENT_ACCUMULATION=8

for j in 3 3
do
  if [ ${MODELS[$j]} == "unified" ]; then
    TRAIN_FILE="/home/katjal/QAslovene/datasets/train.csv"
    VALIDATION_FILE="/home/katjal/QAslovene/datasets/val.csv"
    TEST_FILE="/home/katjal/QAslovene/datasets/test_answered.csv"
  else
    TRAIN_FILE="/home/katjal/QAslovene/datasets/encoded/${MODELS[$j]}/train.csv"
    VALIDATION_FILE="/home/katjal/QAslovene/datasets/encoded/${MODELS[$j]}/val.csv"
    TEST_FILE="/home/katjal/QAslovene/datasets/encoded/${MODELS[$j]}/test_answered.csv"
  fi
  for i in 0 0
  do
  OUTPUT_DIR="/home/katjal/QAslovene/models/${MODELS[$j]}-${MODEL_SHORTHANDS[$i]}"
  MODEL_NAME_OR_PATH=${MODEL_PATHS[$i]}
  CUDA_VISIBLE_DEVICES=1 python run_summarization.py\
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

done