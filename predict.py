from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, Seq2SeqTrainer
import pandas as pd
import torch
import numpy as np
import datasets
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm


# predict("./models/BoolQ/", "./datasets/encoded/BoolQ/test_answered.csv")

dataset_names = ["BoolQ", "COPA", "MCTest", "MultiRC", "SQUAD2"]
model_name_or_path = f"./models/unified-small/"
# Load trained model
config = AutoConfig.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=config)
print("Tokenizer and model loaded")

for dataset_name in dataset_names:
    dataset = datasets.load_dataset("csv", data_files=f"./datasets/encoded/{dataset_name}/test_answered.csv")
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    predictions = list(map(lambda x: x["generated_text"], pipe(dataset["train"]["input"])))
    with open(f"./predictions/{dataset_name}/generated_predictions.txt", "w") as writer:
        writer.write("\n".join(predictions))

