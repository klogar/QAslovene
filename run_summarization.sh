MODEL_PATHS=("/storage/private/student-lkm/t5_sl_small-pytorch" "/storage/private/student-lkm/t5_sl_large-pytorch/")
MODEL_SHORTHANDS=("small" "large")
MODELS=("BoolQ" "COPA" "MCTest" "MultiRC" "SQUAD2" "unified")
MAX_SOURCE_LENGTH=512
MAX_TARGET_LENGTH=100
PER_DEVICE_TRAIN_BATCH_SIZE=8
PER_DEVICE_EVAL_BATCH_SIZE=8
NUM_TRAIN_EPOCHS=25
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
  OUTPUT_DIR="/home/katjal/QAslovene/models/unified-without-noanswer"
  MODEL_NAME_OR_PATH=${MODEL_PATHS[$i]}
  CUDA_VISIBLE_DEVICES=0 python run_summarization.py\
   --model_name_or_path $MODEL_NAME_OR_PATH \
   --no_use_fast_tokenizer \
   --train_file "train.csv" \
   --validation_file "val.csv" \
   --input_column "input" \
   --output_column "output" \
   --max_source_length $MAX_SOURCE_LENGTH \
   --max_target_length $MAX_TARGET_LENGTH \
   --output_dir $OUTPUT_DIR \
   --do_train --do_eval \
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
   --save_total_limit=1 \
   --datasets "SQUAD2,BoolQ,COPA,MCTest,MultiRC" \
   --datasets_path "/home/katjal/QAslovene/datasets/encoded/" \
   --num_beams=4 \
   --filter_no_answer=True
  done

done