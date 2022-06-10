MAX_SOURCE_LENGTH=512
MAX_TARGET_LENGTH=100
PER_DEVICE_EVAL_BATCH_SIZE=8

MODEL_NAME_OR_PATH="allenai/unifiedqa-t5-small"
OUTPUT_DIR="predictions/English-lower"
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
 --metric_for_best_model "predict_rougeL" \
 --greater_is_better=True \
 --save_total_limit=1 \
 --datasets "SQUAD2-eng,BoolQ-eng,COPA-eng,MCTest-eng,MultiRC-eng" \
 --datasets_path "/home/katjal/QAslovene/datasets/encoded/" \
 --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
 --num_beams=4 \
 --lowercase=True

#--num_beams=4 \