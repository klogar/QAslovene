MAX_SOURCE_LENGTH=512
MAX_TARGET_LENGTH=100

MODEL_NAME_OR_PATH="/home/katjal/QAslovene/models/without-BoolQ"
OUTPUT_DIR=$MODEL_NAME_OR_PATH
CHECKPOINTS=($( ls -d ${MODEL_NAME_OR_PATH}/checkpoint* ))
for checkpoint in "${CHECKPOINTS[@]}"
do
  echo "$checkpoint"
  CUDA_VISIBLE_DEVICES=1 python run_summarization.py\
   --model_name_or_path $checkpoint \
   --no_use_fast_tokenizer \
   --test_file "val.csv" \
   --input_column "input" \
   --output_column "output" \
   --max_source_length $MAX_SOURCE_LENGTH \
   --max_target_length $MAX_TARGET_LENGTH \
   --output_dir $checkpoint \
   --do_predict \
   --seed 42 \
   --predict_with_generate\
   --metric_for_best_model "predict_rougeL" \
   --greater_is_better=True \
   --datasets "SQUAD2,BoolQ,COPA,MCTest,MultiRC" \
   --datasets_path "/home/katjal/QAslovene/datasets/encoded/" \
   --num_beams=4
done


