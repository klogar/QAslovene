# read lines and calculate F1/EM
import collections
import string
import re
import argparse
import json
import sys

from collections import Counter
from os import listdir
from os.path import isfile, join, exists
import pandas as pd


def normalize_answer(s):
  """Lower text and remove punctuation and extra whitespace."""
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_punc(lower(s)))

def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
  return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
  gold_toks = get_tokens(a_gold)
  pred_toks = get_tokens(a_pred)
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

def eval_dir(dataset, model):
    test_data = pd.read_csv(f"../datasets/encoded/{dataset}/test_answered.csv")
    gold = [[g] for g in list(test_data["output"])]

    prediction_file = f"../models/{model}/{dataset}_generated_predictions.txt"
    with open(prediction_file) as f:
        predictions = [line.strip() for line in f.readlines()]

    assert len(gold) == len(predictions), f" {len(predictions)}  / {len(gold)} "

    f1 = exact_match = total = 0
    for i, prediction in enumerate(predictions):
        # For unanswerable questions, only correct answer is empty string
        is_unanswerable = False
        for g in gold[i]:
            if no_ans in g.lower():
                # print(gold[i])
                gold[i] = [""]
                is_unanswerable = True
                break

        if no_ans in prediction.lower():
            prediction = ""

        em_current = max(compute_exact(a, prediction) for a in gold[i])
        f1_current = max(compute_f1(a, prediction) for a in gold[i])

        exact_match += em_current
        f1 += f1_current
        total += 1

        if normalize_answer(prediction) not in ["da", "ne"]:
            print(f"Non yes/no prediction: {prediction}")
            print(f"Question:  {test_data['input'][i]}")
        if no_ans in prediction.lower():
            print(prediction)


    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    eval = {'exact_match': exact_match, 'f1': f1}
    print(f" * {dataset} -> {eval}")

no_ans = "< ni odgovora >"

datasets = ["BoolQ"]
model = "unified-general"
for dataset in datasets:
    evaluation = eval_dir(dataset, model)
