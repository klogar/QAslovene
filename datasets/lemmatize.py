import classla
import jsonlines
import pandas as pd
from os import listdir
from os.path import isfile, join
import re


def lemmatize(text):
    return " ".join(nlp(text).get("lemma"))


def lemmatize_predictions(d):
    onlyfiles = [f for f in listdir(d) if isfile(join(d, f)) if f.endswith("test_answered_generated_predictions.txt")]

    for file in onlyfiles:
        filename = file.split(".")[0]
        output = open(f"{d}/{filename}_lemmatized.txt", "w", encoding='UTF8')
        with open(f"{d}/{file}", encoding='UTF8') as f:
            lines = f.readlines()
            for line in lines:
                output.write(lemmatize(line))
                output.write("\n")

# Lemmatize squad2 and multirc which have an array of answers in answers folder
def lemmatize_answers_jsonl(dataset):
    writer = jsonlines.open(f"encoded/lemmatized-answers/{dataset}-test_answered.jsonl", mode="w")

    with jsonlines.open(f"encoded/answers/{dataset}-test_answered.jsonl") as reader:
        golds = [line for line in reader]
        for gold in golds:
            g = list(map(lemmatize, gold["answers"]))
            writer.write({"answers": g})

# Lemmatize mc datasets where we have lemmatize the mc options as well
def lemmatize_mc(dataset):
    writer = jsonlines.open(f"encoded/lemmatized-answers/{dataset}-test_answered.jsonl", mode="w")
    test_file = f"encoded/{dataset}/test_answered.csv"
    test_data = pd.read_csv(test_file)
    answers = list(test_data["output"])
    input_lines = list(test_data["input"])

    regex = re.compile("\([A-E]\)")

    for answer, input in zip(answers, input_lines):
        ans = lemmatize(answer)
        input_split = input.split("\\n")
        candidates_string = input_split[1].strip()
        candidates_split = regex.split(candidates_string)
        candidates_split = [lemmatize(x.strip()) for x in candidates_split if len(x.strip()) > 0]

        writer.write({"answers": ans, "options": candidates_split})

# Lemmatize boolq
def lemmatize_test_answered(dataset):
    writer = jsonlines.open(f"encoded/lemmatized-answers/{dataset}-test_answered.jsonl", mode="w")
    test_file = f"encoded/{dataset}/test_answered.csv"
    test_data = pd.read_csv(test_file)
    for answer in test_data["output"]:
        ans = lemmatize(answer)
        writer.write(ans)


classla.download("sl")
nlp = classla.Pipeline("sl", processors="tokenize,pos,lemma")

lemmatize_answers_jsonl("MultiRC")
# lemmatize_answers_jsonl("SQUAD2-project")
# lemmatize_test_answered("BoolQ")
# lemmatize_mc("COPA")
# lemmatize_mc("MCTest")
# lemmatize_predictions("../models/unified-lower-all/checkpoint-361460")

