# read lines and calculate F1/EM
import collections
import string
import re
import argparse
import json
import sys
import rouge
import copy

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

rouge_l_evaluator = rouge.Rouge(
    metrics=["rouge-l"],
    max_n=4,
    limit_length=True,
    length_limit=100,
    length_limit_type="words",
    apply_avg=True,
    apply_best=True,
    alpha=0.5,
    weight_factor=1.2,
    stemming=True,
)

def rouge_l(p, g):
    return rouge_l_evaluator.get_scores(p, g)

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, [ground_truth])
        scores_for_ground_truths.append(score)
    if isinstance(score, dict) and "rouge-l" in score:
        max_score = copy.deepcopy(score)
        max_score["rouge-l"]["f"] = round(
            max([score["rouge-l"]["f"] for score in scores_for_ground_truths]), 2
        )
        max_score["rouge-l"]["p"] = round(
            max([score["rouge-l"]["p"] for score in scores_for_ground_truths]), 2
        )
        max_score["rouge-l"]["r"] = round(
            max([score["rouge-l"]["r"] for score in scores_for_ground_truths]), 2
        )
        return max_score
    else:
        return round(max(scores_for_ground_truths), 2)

def eval_dir(model):
    test_data = pd.read_csv("../datasets/encoded/SQUAD2/test_answered.csv")
    gold = [[g] for g in list(test_data["output"])]

    prediction_file = f"../models/{model}/SQUAD2_generated_predictions.txt"
    with open(prediction_file) as f:
        predictions = [line.strip() for line in f.readlines()]

    assert len(gold) == len(predictions), f" {len(predictions)}  / {len(gold)} "

    f1 = exact_match = total = 0
    f1_with_ans = exact_match_with_ans = total_with_ans = 0
    f1_wout_ans = exact_match_wout_ans = total_wout_ans = 0
    scores = []
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

        if is_unanswerable:
            exact_match_wout_ans += em_current
            f1_wout_ans += f1_current
            total_wout_ans += 1
        else:
            exact_match_with_ans += em_current
            f1_with_ans += f1_current
            total_with_ans += 1

        # Print some answers and wrong predictions
        # if em_current == 0 and i < 100:
        #     print(f"Prediction: {prediction}")
        #     print(f"True: {gold[i]}")

        # Also calculate rouge L
        rouge_l_score = metric_max_over_ground_truths(rouge_l, prediction, gold[i])
        scores.append(rouge_l_score["rouge-l"]["f"])

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    exact_match_with_ans = 100.0 * exact_match_with_ans / total_with_ans
    f1_with_ans = 100.0 * f1_with_ans / total_with_ans
    exact_match_wout_ans = 100.0 * exact_match_wout_ans / total_wout_ans
    f1_wout_ans = 100.0 * f1_wout_ans / total_wout_ans
    rougeL = 100.0 * sum(scores) / len(scores)
    eval = {'exact_match': exact_match, 'f1': f1,
            'exact_match with answers': exact_match_with_ans, 'f1 with answers': f1_with_ans,
            'exact_match without answers': exact_match_wout_ans, 'f1 without answers': f1_wout_ans,
            'rouge L': rougeL}

    print(f" * SQUAD2 -> {eval}")

def eval_dir_orig(dir, checkpoint='all'):
    # print(dir)
    all_predictions = []
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]

    gold = []
    with open(gold_all_ans) as f:
        for l in f.readlines():
            gold.append(json.loads(l.replace("\n", "")))

    only_predictions = [f for f in onlyfiles if "_predictions" in f]
    if checkpoint == 'all':
        only_predictions = sorted(only_predictions)
    else:
        only_predictions = [x for x in only_predictions if
                        int(x.split('_')[-2]) > 1090500 and int(x.split('_')[-2]) < 1110000]

    for file in only_predictions:
        print(dir + file)
        predictions = []
        with open(dir + file) as f:
            for l in f.readlines():
                predictions.append(l.replace("\n", ""))

        assert len(gold) == len(predictions), f" {len(predictions)}  / {len(gold)} "


        f1 = exact_match = total = 0
        for i, prediction in enumerate(predictions):
            # For unanswerable questions, only correct answer is empty string
            for g in gold[i]:
                if no_ans in g.lower():
                    # print(gold[i])
                    gold[i] = [""]
                    break

            if no_ans in prediction.lower():
                prediction = ""

            exact_match += max(compute_exact(a, prediction) for a in gold[i])
            f1 += max(compute_f1(a, prediction) for a in gold[i])

            total += 1

        exact_match = 100.0 * exact_match / total
        f1 = 100.0 * f1 / total
        eval = {'exact_match': exact_match, 'f1': f1}
        print(f" * {dir}{file} -> {eval}")
        all_predictions.append([file, eval])
    return all_predictions

no_ans = "< ni odgovora >"
model = "unified-general"
evaluation = eval_dir(model)
