MODEL_PATHS=("/storage/private/student-lkm/t5_sl_small-pytorch" "/storage/private/student-lkm/t5_sl_large-pytorch/")
MODEL_SHORTHANDS=("small" "large")
MODELS=("BoolQ" "COPA" "MCTest" "MultiRC" "SQUAD2" "unified")
MAX_SOURCE_LENGTH=512
MAX_TARGET_LENGTH=100
PER_DEVICE_TRAIN_BATCH_SIZE=8
PER_DEVICE_EVAL_BATCH_SIZE=8
NUM_TRAIN_EPOCHS=150
GRADIENT_ACCUMULATION=1

for j in 5 5
do
  if [ ${MODELS[$j]} == "unified" ]; then
    TRAIN_FILE="/home/katjal/QAslovene/datasets/train.csv"
    VALIDATION_FILE="/home/katjal/QAslovene/datasets/val.csv"
#    TEST_FILE="/home/katjal/QAslovene/datasets/test_answered.csv"
  else
    TRAIN_FILE="/home/katjal/QAslovene/datasets/encoded/${MODELS[$j]}/train.csv"
    VALIDATION_FILE="/home/katjal/QAslovene/datasets/encoded/${MODELS[$j]}/val.csv"
#    TEST_FILE="/home/katjal/QAslovene/datasets/encoded/${MODELS[$j]}/test_answered.csv"
  fi
  for i in 0 0
  do
  #OUTPUT_DIR="/home/katjal/QAslovene/models/${MODELS[$j]}-${MODEL_SHORTHANDS[$i]}"
  MODEL_NAME_OR_PATH="/home/katjal/QAslovene/models/unified-general"
  OUTPUT_DIR=$MODEL_NAME_OR_PATH
  CUDA_VISIBLE_DEVICES=1 python run_summarization.py\
   --model_name_or_path $MODEL_NAME_OR_PATH \
   --no_use_fast_tokenizer \
   --test_file "test_answered.csv" \
   --input_column "input" \
   --output_column "output" \
   --max_source_length $MAX_SOURCE_LENGTH \
   --max_target_length $MAX_TARGET_LENGTH \
   --output_dir $OUTPUT_DIR \
   --do_predict \
   --seed 42 \
   --predict_with_generate\
   --metric_for_best_model "eval_rougeL" \
   --greater_is_better=True \
   --save_total_limit=1 \
   --datasets "SQUAD2,BoolQ,COPA,MCTest,MultiRC" \
   --datasets_path "/home/katjal/QAslovene/datasets/encoded/" \
   --num_beams=4

  done

done