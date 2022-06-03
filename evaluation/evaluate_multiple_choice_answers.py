import re
from os import listdir
from os.path import isfile, join, exists
import numpy as np
import pandas as pd
import string
import classla

def score_string_similarity(str1, str2, lemmatized=False):
    if str1 == str2:
        return 3.0   # Better than perfect token match
    str1 = normalize_answer(str1, lemmatized)
    str2 = normalize_answer(str2, lemmatized)
    if str1 == str2:
        return 2.0
    if " " in str1 or " " in str2:
        str1_split = str1.split(" ")
        str2_split = str2.split(" ")
        overlap = list(set(str1_split) & set(str2_split))
        return len(overlap) / max(len(str1_split), len(str2_split))
    else:
        if str1 == str2:
            return 1.0
        else:
            return 0.0

def normalize_answer(s, lemmatized=False):
  """Lower text and remove punctuation and extra whitespace."""
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  def lemmatize(text):
      if lemmatized:
          return " ".join(nlp(s).get("lemma"))
      return text
  return white_space_fix(remove_punc(lower(lemmatize(s))))


def eval_dir_orig(dir, checkpoints = 'all'):
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]

    only_predictions = [f for f in onlyfiles if "_predictions" in f]
    if checkpoints == 'all':
        only_predictions = sorted(only_predictions)
        # print(only_predictions)
    else:
        only_predictions = [x for x in only_predictions if
                            '.csv' not in x and int(x.split('_')[-2]) > 1090500 and int(x.split('_')[-2]) < 1110000]

        if len(only_predictions) > 1:
            raise EnvironmentError


    for p in only_predictions:
        convert_predictions(input_file, dir + "/" + p, meta_file)

def eval_dir(dataset, model, lemmatized=False):
    convert_predictions(dataset, model, lemmatized)

def convert_predictions(dataset, model, lemmatized=False):
    # predictions_file = f"../predictions/{name}/generated_predictions-150.txt" # unified common model
    predictions_file = f"../models/{model}/{dataset}_generated_predictions.txt" # only trained on its own dataset
    test_file = f"../datasets/encoded/{dataset}/test_answered.csv"

    test_data = pd.read_csv(test_file)
    gold_lines = list(test_data["output"])
    input_lines = list(test_data["input"])

    with open(predictions_file) as f:
        predictions = [line.strip() for line in f.readlines()]

    assert len(gold_lines) == len(predictions)


    accuracy = []
    for prediction, input, gold in zip(predictions, input_lines, gold_lines):
        input_split = input.split("\\n")
        candidates_string = input_split[1].strip()
        candidates_split = regex.split(candidates_string)
        candidates_split = [x.strip() for x in candidates_split if len(x.strip()) > 0]
        # print(f"{prediction} <-> {candidates_split}")
        scores = [score_string_similarity(x, prediction, lemmatized) for x in candidates_split]
        max_idx = np.argmax(scores)
        # TODO: If multiple options has max score, look for best token alignment
        selected_ans = candidates_split[max_idx]

        # print((gold, selected_ans))
        if normalize_answer(selected_ans) == normalize_answer(gold):
            accuracy.append(1)
        else:
            accuracy.append(0)

        if max(scores) == 0:
            print(f" ***** ERRROR: {prediction} <-> {candidates_split} ")
            # break

    print(f" *** {dataset} \t {100.0 * sum(accuracy) / len(accuracy)}")

def convert_predictions_orig(input, pred, meta_file):
    input_lines = []
    with open(input) as f:
        for line in f.readlines():
            input_lines.append(line.split("\t")[0])
            # print(line.split("\t")[0])

    with open(pred) as f:
        pred_lines = list(f.readlines())

    with open(meta_file) as f:
        id_lines = list(f.readlines())

    if len(pred_lines) != len(input_lines):
        print("skipping . . . ")
        return

    assert len(pred_lines) == len(input_lines), f"{len(pred_lines)} vs {len(input_lines)} / {input} / {pred}"

    outfile = open(pred + "_selected_candidate.csv", "w")

    accuracy = []
    for prediction, input, id_and_gold in zip(pred_lines, input_lines, id_lines):
        prediction = prediction.replace("\\n", "").strip()
        id_and_gold_split = id_and_gold.replace("\\n", "").strip().split("\t")
        id = id_and_gold_split[0]
        gold = id_and_gold_split[1]
        assert len(gold) < 3, f"gold: {gold} - id_and_gold: {id_and_gold} "
        is_numeric = False
        numeric_from_zero = False
        if len(id_and_gold_split) > 2:
            if "numeric" in id_and_gold_split[2]:
                is_numeric = True
            if "numeric_from_zero" in id_and_gold_split[2]:
                numeric_from_zero = True
        # print((is_numeric, numeric_from_zero))
        input_split = input.split("\\n")
        # print(input_split)
        candidates_string = input_split[1].strip().lower()
        candidates_split = regex.split(candidates_string)
        candidates_split = [x.strip() for x in candidates_split if len(x.strip()) > 0]
        # print(f"{prediction} <-> {candidates_split}")
        scores = [score_string_similarity(x, prediction) for x in candidates_split]
        max_idx = np.argmax(scores)
        # TODO: If multiple options has max score, look for best token alignment
        selected_ans = chr(ord('A') + max_idx)

        # print((gold, selected_ans))
        if selected_ans == gold:
            accuracy.append(1)
        else:
            accuracy.append(0)

        if is_numeric:
            # print(f"is numeric: {selected_ans} -> {chr(ord('1') + max_idx)}")
            if numeric_from_zero:
                selected_ans = chr(ord('0') + max_idx)
            else:
                selected_ans = chr(ord('1') + max_idx)

        outfile.write(f"{id},{selected_ans}\n")

        # if max(scores) == 0:
        #     print(f" ***** ERRROR: {prediction} <-> {candidates_split} ")
            # break

    print(f" *** {pred} \t {100.0 * sum(accuracy) / len(accuracy)}")

regex = re.compile("\([A-E]\)")
datasets = ["MCTest", "COPA"]
model = "unified-general"
lemmatized = True
if lemmatized:
    classla.download("sl")
    nlp = classla.Pipeline("sl", processors="tokenize,pos,lemma")
for dataset in datasets:
    evaluation = eval_dir(dataset, model, lemmatized)








