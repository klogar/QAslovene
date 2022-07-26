from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, Seq2SeqTrainer
import pandas as pd
import datasets

def generate(model):
    model_name_or_path = f"./models/{model}/"

    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=config)
    print(f"Tokenizer and model loaded from {model_name_or_path}")

    for dataset in datasets:
        input = list(pd.read_csv(f"./datasets/encoded/{dataset}/{kind}.csv")["input"])

        tokenized_input = tokenizer(input, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
        # predictions = model.generate(**tokenized_input, max_length=100, num_beams=4, early_stopping=True, no_repeat_ngram_size=3)
        predictions = model.generate(**tokenized_input, max_length=100)
        predictions_text = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        with open(f"{model_name_or_path}/{dataset}_{kind}_predictions_greedy.txt", "w") as writer:
            writer.write("\n".join(predictions_text))

datasets = ["BoolQ", "COPA", "MCTest", "MultiRC", "SQUAD2"]
lowercase = False
input_column = "input"
output_column = "output"
max_source_length = 512
max_target_length = 100
kind = "test_answered"

generate("unified-general-all")