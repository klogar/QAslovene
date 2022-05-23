import copy
import json
import rouge
from os import listdir
from os.path import isfile, join, exists
import jsonlines
import string
import collections

# from rouge_score import rouge

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
    with jsonlines.open(f"./../datasets/encoded/answers/MultiRC.jsonl") as reader:
        golds = [line["answers"] for line in reader]

    prediction_file = f"../models/{model}/MultiRC_generated_predictions.txt"
    with open(prediction_file) as f:
        predictions = [line.strip() for line in f.readlines()]

    scores = []
    scores_em_bool = []
    scores_f1_bool = []
    total_bool = 0
    total_no_ans = 0

    for gold, pred in zip(golds, predictions):
        rouge_l_score = metric_max_over_ground_truths(rouge_l, pred, gold)
        scores.append(rouge_l_score["rouge-l"]["f"])
        normalized_gold = list(map(normalize_answer, gold))
        if "da" in normalized_gold or "ne" in normalized_gold:
            # if normalize_answer(pred) != "da" and normalize_answer(pred) != "ne":
            #     print(pred, gold)
            em_current = max(compute_exact(a, pred) for a in gold)
            f1_current = max(compute_f1(a, pred) for a in gold)
            scores_em_bool.append(em_current)
            scores_f1_bool.append(f1_current)
            total_bool += 1
        if no_ans in pred.lower() and no_ans not in gold:
            total_no_ans += 1
            # print(pred, gold)

    print(f" * MultiRC -> rouge L {100.0 * sum(scores) / len(scores)}, em for bool: {100.0 * sum(scores_em_bool) / total_bool}, f1 for bool: {100.0 * sum(scores_f1_bool) / total_bool}, no answer: {total_no_ans/len(scores)}")

def eval_dir_orig(dir, checkpoints='all'):
    all_predictions = []
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
    only_targets = [f for f in onlyfiles if "_targets" in f]
    if len(only_targets) > 1:
        return None
    assert len(only_targets) == 1, f"targets: {only_targets} - dir: {dir}"

    only_inputs = [f for f in onlyfiles if "inputs" in f]
    if len(only_inputs) > 1:
        return None
    assert len(only_inputs) == 1, f"inputs: {only_inputs} - dir: {dir}"

    instances = {}
    with open(dir + only_inputs[0]) as f:
        for i, l in enumerate(f.readlines()):
            if l not in instances:
                instances[l] = []
            instances[l].append(i)

    golds = []
    with open(dir + only_targets[0]) as f:
        for l in f.readlines():
            golds.append(l.replace("\\n", "").strip())

    only_predictions = [f for f in onlyfiles if "_predictions" in f]
    if checkpoints == 'all':
        only_predictions = sorted(only_predictions)
    else:
        only_predictions = [x for x in only_predictions if
                            int(x.split('_')[-2]) > 1090500 and int(x.split('_')[-2]) < 1110000]

    for file in only_predictions:
        predictions = []
        with open(dir + file) as f:
            for l in f.readlines():
                predictions.append(l.replace("\\n", "").strip())

        assert len(golds) == len(predictions), f" {len(predictions)}  / {len(golds)} "
        scores = []
        for k, v in instances.items():
            golds_subset = [golds[i] for i in v]
            rouge_l_score = metric_max_over_ground_truths(rouge_l, predictions[v[0]], golds_subset)
            scores.append(rouge_l_score["rouge-l"]["f"])

        print(f" * {dir}{file} -> {100.0 * sum(scores) / len(scores)}")

        all_predictions.append([file, eval])
    return all_predictions

no_ans = "< ni odgovora >"
model = "unified-general"
evaluation = eval_dir(model)
