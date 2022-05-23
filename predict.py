from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, Seq2SeqTrainer
import pandas as pd
import torch
import numpy as np
import datasets
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm


# predict("./models/BoolQ/", "./datasets/encoded/BoolQ/test_answered.csv")


# Load trained model
def predict(model, dataset_names, prediction_filename):
    model_name_or_path = f"./models/{model}/"

    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=config)
    print(f"Tokenizer and model loaded from {model_name_or_path}")

    for dataset_name in dataset_names:
        dataset = datasets.load_dataset("csv", data_files={"test": f"./datasets/encoded/{dataset_name}/test_answered.csv"})
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

        predictions = list(map(lambda x: x["generated_text"], pipe(dataset["test"]["input"])))
        with open(f"./predictions/{dataset_name}/{prediction_filename}.txt", "w") as writer:
            writer.write("\n".join(predictions))


dataset_names = ["BoolQ", "COPA", "MCTest", "MultiRC", "SQUAD2"]
# dataset_names = ["MultiRC"]
models = ["BoolQ-small", "COPA-small", "MCTest-small", "MultiRC-small", "SQUAD2-small", "unified-small"]

for model in models[-1:]:
    predict(model, dataset_names, f"generated_predictions_dev")

