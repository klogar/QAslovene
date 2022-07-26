from os import listdir
from os.path import isfile, join, exists, isdir
import pandas as pd
from eval_utils_eng import *
import re
import numpy as np
import classla
import jsonlines
import matplotlib.pyplot as plt


def eval_bool(dataset, model):
    print(f"--------------------------{dataset}--------------------------------")
    test_data = pd.read_csv(f"../datasets/encoded/{dataset}/{kind}.csv")
    gold = [[g] for g in list(test_data["output"])]
    context = [input.split("\\n")[1] for input in list(test_data["input"])]
    type = list(test_data["type"])

    dir = f"../predictions/{folder}"
    if checkpoint == "all":
        all_directories = [f"{dir}/{f}" for f in listdir(dir) if isdir(join(dir, f)) and "checkpoint-" in f]
        all_directories.sort(key=lambda k: (int(k.split("-")[-1])))
    elif checkpoint is not None:
        all_directories = [f"{dir}/checkpoint-{checkpoint}"]
    else:
        all_directories = [dir]

    evals = []

    for directory in all_directories:
        if kind_in_prediction_file:
            prediction_file = f"{directory}/{dataset}_{kind}_generated_predictions.txt"
        else:
            prediction_file = f"{directory}/{dataset}_generated_predictions.txt"
        with open(prediction_file) as f:
            predictions = [line.strip() for line in f.readlines()]

        assert len(gold) == len(predictions), f" {len(predictions)}  / {len(gold)} "

        f1 = exact_match = total = 0
        answer_is_start_of_context = 0
        total_no_ans = 0
        total_not_yesno = 0
        for i, prediction in enumerate(predictions):
            if filter_machine_translation and type[i] == "MT":
                continue
            # For unanswerable questions, only correct answer is empty string
            is_unanswerable = False
            for g in gold[i]:
                if no_ans in g.lower():
                    # print(gold[i])
                    gold[i] = [""]
                    is_unanswerable = True
                    break
                if g == "": # when we already converted <no answer> into empty string
                    is_unanswerable = True
            if no_ans in prediction.lower():
                prediction = ""

            em_current = max(compute_exact(a, prediction) for a in gold[i])
            f1_current = max(compute_f1(a, prediction) for a in gold[i])

            exact_match += em_current
            f1 += f1_current
            total += 1

            if prediction == "": #no answer
                total_no_ans += 1

            if verbose:
                if normalize_answer(prediction) not in ["yes", "no"]:
                    print(f"Non yes/no prediction: {prediction}")
                    print(f"Question:  {test_data['input'][i]}")
                    total_not_yesno += 1
                if no_ans in prediction.lower():
                    print(prediction)
                if normalize_answer(context[i]).startswith(normalize_answer(prediction)):
                    answer_is_start_of_context += 1
                # else:
                #     print(prediction)
                #     print(context[i])

        if verbose:
            print(f"answer is start of context {answer_is_start_of_context/total}")
        print(f"predicted no answer: {total_no_ans/total}")
        print(f"Not yes no {total_not_yesno/total} ")


        exact_match = round(exact_match / total, 3)
        f1 = round(f1 / total, 3)

        eval = {'exact_match': exact_match, 'f1': f1}
        # print(f" * {dataset} -> {eval}")
        evals.append(eval)
        print()
    print(f"-------------------------end {dataset}---------------------------------")
    print()
    return evals

def eval_mc(dataset, model):
    print(f"--------------------------{dataset}--------------------------------")
    test_file = f"../datasets/encoded/{dataset}/{kind}.csv"

    dir = f"../predictions/{folder}"
    if checkpoint == "all":
        all_directories = [f"{dir}/{f}" for f in listdir(dir) if isdir(join(dir, f)) and "checkpoint-" in f]
        all_directories.sort(key=lambda k: (int(k.split("-")[-1])))
    elif checkpoint is not None:
        all_directories = [f"{dir}/checkpoint-{checkpoint}"]
    else:
        all_directories = [dir]

    evals = []

    for directory in all_directories:
        if kind_in_prediction_file:
            prediction_file = f"{directory}/{dataset}_{kind}_generated_predictions.txt"
        else:
            prediction_file = f"{directory}/{dataset}_generated_predictions.txt"

        test_data = pd.read_csv(test_file)
        gold_lines = list(test_data["output"])
        input_lines = list(test_data["input"])
        type_lines = list(test_data["type"])

        with open(prediction_file) as f:
            predictions = [line.strip() for line in f.readlines()]

        assert len(gold_lines) == len(predictions)

        regex = re.compile("\([A-E]\)")

        accuracy = []
        similar_to_context = 0
        similar_to_quesiton = 0
        completely_wrong = 0
        total_no_ans = 0
        scores_em = []
        for ind, (prediction, input, gold) in enumerate(zip(predictions, input_lines, gold_lines)):
            if filter_machine_translation and type_lines[ind] == "MT":
                continue
            input_split = input.split("\\n")
            candidates_string = input_split[1].strip()
            candidates_split = regex.split(candidates_string)
            candidates_split = [x.strip() for x in candidates_split if len(x.strip()) > 0]
            # print(f"{prediction} <-> {candidates_split}")
            scores = [score_string_similarity(x, prediction) for x in candidates_split]
            max_idx = np.argmax(scores)
            if max(scores) == 0:
                accuracy.append(0)
                completely_wrong += 1
                if verbose:
                    print(f" ***** ERRROR: {prediction} <-> {candidates_split} ")
            else:
                # TODO: If multiple options has max score, look for best token alignment
                selected_ans = candidates_split[max_idx]

                # print((gold, selected_ans), candidates_split)
                if normalize_answer(selected_ans) == normalize_answer(gold):
                    accuracy.append(1)
                    # print(selected_ans, gold)
                else:
                    accuracy.append(0)
            # exact match
            em = compute_exact(gold, prediction)
            scores_em.append(em)


            context = input_split[2]
            question = input_split[0]
            if score_string_similarity(context, prediction) > max(scores):
                similar_to_context += 1
            if score_string_similarity(question, prediction) > max(scores):
                similar_to_quesiton += 1
                # print(f"{accuracy[-1]} <-> {prediction} <-> {gold} <-> {candidates_split} <-> {context}")

            if no_ans in prediction.lower():
                total_no_ans += 1

        print(f"similar to question {similar_to_quesiton/len(accuracy)}")
        print(f"similar to context {similar_to_context/len(accuracy)}")
        print(f"completely wrong {completely_wrong/len(accuracy)}")
        print(f"predicted no answer {total_no_ans/len(accuracy)}")



        accuracy = round(sum(accuracy) / len(accuracy), 3)
        em_score = round(sum(scores_em) / len(scores_em), 3)
        eval = {'accuracy': accuracy, 'exact_match': em_score}
        evals.append(eval)
        print()
    print(f"--------------------------end {dataset}--------------------------------")
    print()
    return evals

def eval_multirc(dataset, model):
    print(f"--------------------------{dataset}--------------------------------")
    with jsonlines.open(f"./../datasets/encoded/answers/MultiRC-eng-{kind}.jsonl") as reader:
        golds = [line["answers"] for line in reader]
    test_file = f"../datasets/encoded/{dataset}/{kind}.csv"

    dir = f"../predictions/{folder}"
    if checkpoint == "all":
        all_directories = [f"{dir}/{f}" for f in listdir(dir) if isdir(join(dir, f)) and "checkpoint-" in f]
        all_directories.sort(key=lambda k: (int(k.split("-")[-1])))
    elif checkpoint is not None:
        all_directories = [f"{dir}/checkpoint-{checkpoint}"]
    else:
        all_directories = [dir]

    evals = []

    for directory in all_directories:
        if kind_in_prediction_file:
            prediction_file = f"{directory}/{dataset}_{kind}_generated_predictions.txt"
        else:
            prediction_file = f"{directory}/{dataset}_generated_predictions.txt"

        with open(prediction_file) as f:
            predictions = [line.strip() for line in f.readlines()]

        test_data = pd.read_csv(test_file)
        type_lines = list(test_data["type"])

        scores = []
        scores_em = []
        scores_f1 = []
        scores_em_bool = []
        scores_f1_bool = []
        total_bool = total_yes = total_no = 0
        total_yes_gold = total_no_gold = 0
        total_no_ans = 0

        for ind, (gold, pred) in enumerate(zip(golds, predictions)):
            if no_ans in gold:
                continue # skip no answer questions
            if filter_machine_translation and type_lines[ind] == "MT":
                continue
            rouge_l_score = metric_max_over_ground_truths(rouge_l, pred, gold)
            scores.append(rouge_l_score["rouge-l"]["f"])
            normalized_gold = list(map(normalize_answer, gold))

            em_current = max(compute_exact(a, pred) for a in gold)
            f1_current = max(compute_f1(a, pred) for a in gold)
            scores_em.append(em_current)
            scores_f1.append(f1_current)

            if "yes" in normalized_gold or "no" in normalized_gold:
                # if normalize_answer(pred) != "da" and normalize_answer(pred) != "ne":
                #     print(pred, gold)

                scores_em_bool.append(em_current)
                scores_f1_bool.append(f1_current)
                total_bool += 1
                if normalize_answer(pred) == "no":
                    total_no += 1
                elif normalize_answer(pred) == "yes":
                    total_yes += 1
                if "yes" in normalized_gold:
                    total_yes_gold += 1
                if "no" in normalized_gold:
                    total_no_gold += 1
                # print(f"{pred} || {gold}")
            # if normalize_answer(pred) == "ne" or normalize_answer(pred) == "da":
            #     print(f"{pred} || {gold}")
            if no_ans in pred.lower() and no_ans not in gold:
                total_no_ans += 1
                # print(pred, gold)

        print(total_bool, len(scores), total_bool/len(scores))
        print(f"yes: {total_yes/total_bool}, no: {total_no/total_bool}, total_yes: {total_yes_gold/total_bool}, total_no: {total_no_gold/total_bool}")
        rougeL_score = round(sum(scores) / len(scores), 3)
        em_score = round(sum(scores_em) / len(scores), 3)
        f1_score = round(sum(scores_f1) / len(scores), 3)
        em_bool_score = round(sum(scores_em_bool) / total_bool, 3)
        f1_bool_score = round(sum(scores_f1_bool) / total_bool, 3)
        no_answer = round(total_no_ans/len(scores), 3)
        eval = {"rougeL": rougeL_score,
                "em": em_score,
                "f1": f1_score,
                "em_bool": em_bool_score,
                "f1_bool": f1_bool_score,
                "no_answer": no_answer}
        evals.append(eval)
        # print()
    print(f"--------------------------end {dataset}--------------------------------")
    print()
    return evals

def eval_squad2(dataset, model):
    print(f"--------------------------{dataset}--------------------------------")
    with jsonlines.open(f"./../datasets/encoded/answers/SQUAD2-eng-{kind}.jsonl") as reader:
        gold = [line["answers"] for line in reader]
    test_file = f"../datasets/encoded/{dataset}/{kind}.csv"

    dir = f"../predictions/{folder}"
    if checkpoint == "all":
        all_directories = [f"{dir}/{f}" for f in listdir(dir) if isdir(join(dir, f)) and "checkpoint-" in f]
        all_directories.sort(key=lambda k: (int(k.split("-")[-1])))
    elif checkpoint is not None:
        all_directories = [f"{dir}/checkpoint-{checkpoint}"]
    else:
        all_directories = [dir]

    evals = []

    for directory in all_directories:
        if kind_in_prediction_file:
            prediction_file = f"{directory}/{dataset}_{kind}_generated_predictions.txt"
        else:
            prediction_file = f"{directory}/{dataset}_generated_predictions.txt"
        with open(prediction_file) as f:
            predictions = [line.strip() for line in f.readlines()]

        assert len(gold) == len(predictions), f" {len(predictions)}  / {len(gold)} "

        test_data = pd.read_csv(test_file)
        type_lines = list(test_data["type"])

        f1 = exact_match = total = 0
        f1_with_ans = exact_match_with_ans = total_with_ans = 0
        f1_wout_ans = exact_match_wout_ans = total_wout_ans = 0
        total_no_ans = 0
        scores = []
        for i, prediction in enumerate(predictions):
            if filter_machine_translation and type_lines[i] == "MT":
                continue
            # For unanswerable questions, only correct answer is empty string
            is_unanswerable = False
            for g in gold[i]:
                if no_ans in g.lower():
                    # print(gold[i])
                    gold[i] = [""]
                    is_unanswerable = True
                    break
                if g == "": # when we already converted <ni odgovora> into empty string
                    is_unanswerable = True

            if no_ans in prediction.lower():
                prediction = ""

            if filter_no_answer and is_unanswerable:
                continue # skip <no answer> question
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

            if prediction == "" and "" not in gold[i]:
                total_no_ans += 1

            # Print some answers and wrong predictions
            # if em_current == 0 and i < 100:
            #     print(f"Prediction: {prediction}")
            #     print(f"True: {gold[i]}")

            # Also calculate rouge L
            rouge_l_score = metric_max_over_ground_truths(rouge_l, prediction, gold[i])
            scores.append(rouge_l_score["rouge-l"]["f"])

        exact_match = round(exact_match / total, 3)
        f1 = round(f1 / total, 3)
        exact_match_with_ans = round(exact_match_with_ans / total_with_ans, 3)
        f1_with_ans = round(f1_with_ans / total_with_ans, 3)
        exact_match_wout_ans = round(exact_match_wout_ans / total_wout_ans, 3) if total_wout_ans > 0 else 0
        f1_wout_ans = round(f1_wout_ans / total_wout_ans, 3) if total_wout_ans > 0 else 0
        rougeL = round(sum(scores) / len(scores), 3)
        no_answer = round(total_no_ans/len(scores), 3)
        eval = {'exact_match': exact_match, 'f1': f1,
                'exact_match with answers': exact_match_with_ans, 'f1 with answers': f1_with_ans,
                'exact_match without answers': exact_match_wout_ans, 'f1 without answers': f1_wout_ans,
                'rougeL': rougeL, "no_answer": no_answer }
        evals.append(eval)
        # print()
    print(f"--------------------------end {dataset}--------------------------------")
    print()
    return evals

def plot_checkpoints(evaluation):
    epochs = range(len(evaluation["BoolQ"]))
    plt.plot(epochs, list(map(lambda x: x["exact_match"], evaluation["BoolQ"])), label="BoolQ - točnost")
    plt.plot(epochs, list(map(lambda x: x["accuracy"], evaluation["COPA"])), label="COPA - točnost")
    plt.plot(epochs, list(map(lambda x: x["accuracy"], evaluation["MCTest"])), label="MCTest - točnost")
    plt.plot(epochs, list(map(lambda x: x["rougeL"], evaluation["MultiRC"])), label="MultiRC - rougeL")
    plt.plot(epochs, list(map(lambda x: x["f1"], evaluation["SQUAD2"])), label="SQUAD2 - F1")
    plt.axvline(x=best_epoch)
    plt.legend()
    plt.xlabel("Epohe")
    plt.ylabel("Metrike")
    plt.show()

def plot_squad2(evaluation):
    epochs = range(len(evaluation["SQUAD2"]))
    plt.plot(epochs, list(map(lambda x: x["f1"], evaluation["SQUAD2"])), label="SQUAD2 - F1")
    plt.plot(epochs, list(map(lambda x: x["exact_match"], evaluation["SQUAD2"])), label="SQUAD2 - EM")
    plt.plot(epochs, list(map(lambda x: x["exact_match with answers"], evaluation["SQUAD2"])), label="SQUAD2 - EM answers")
    plt.plot(epochs, list(map(lambda x: x["f1 with answers"], evaluation["SQUAD2"])), label="SQUAD2 - F1 answers")
    plt.plot(epochs, list(map(lambda x: x["exact_match without answers"], evaluation["SQUAD2"])), label="SQUAD2 - EM without answers")
    plt.plot(epochs, list(map(lambda x: x["rougeL"], evaluation["SQUAD2"])), label="SQUAD2 - rougeL")
    plt.plot(epochs, list(map(lambda x: x["no_answer"], evaluation["SQUAD2"])), label="SQUAD2 - % of no answer")
    plt.legend()
    plt.xlabel("Epohe")
    plt.ylabel("Metrike")
    plt.show()

def plot_multirc(evaluation):
    epochs = range(len(evaluation["MultiRC"]))
    plt.plot(epochs, list(map(lambda x: x["rougeL"], evaluation["MultiRC"])), label="MultiRC - rougeL")
    plt.plot(epochs, list(map(lambda x: x["em_bool"], evaluation["MultiRC"])), label="MultiRC - EM bool")
    plt.plot(epochs, list(map(lambda x: x["f1_bool"], evaluation["MultiRC"])), label="MultiRC - F1 bool")
    plt.plot(epochs, list(map(lambda x: x["no_answer"], evaluation["MultiRC"])), label="MultiRC - % of no answer")
    plt.legend()
    plt.xlabel("Epohe")
    plt.ylabel("Metrike")
    plt.show()

def get_best_epoch(evaluation):
    average_evals = [0] * len(list(evaluation.values())[0])
    for dataset, evals in evaluation.items():
        # print(f"*** {dataset} *** -> {evals}")
        for ind, eval in enumerate(evals):
            if dataset == "BoolQ":
                average_evals[ind] += eval["exact_match"]
            elif dataset == "COPA" or dataset == "MCTest":
                average_evals[ind] += eval["accuracy"]
            elif dataset == "MultiRC":
                average_evals[ind] += eval["rougeL"]
            elif dataset == "SQUAD2":
                average_evals[ind] += eval["f1"]
    average_evals = [evals / len(evaluation.items()) for evals in average_evals]
    # print(average_evals)
    return average_evals.index(max(average_evals))

no_ans = "no answer>"
model = "allenai/unifiedqa-t5-small"
verbose = True
checkpoint = None # specific checkpoint, "all" or None
kind = "test_answered"
kind_in_prediction_file = True
filter_no_answer = False
filter_machine_translation = False
folder = "English-lower"

evaluation = dict()
evaluation["BoolQ"] = eval_bool("BoolQ-eng", model)
evaluation["COPA"] = eval_mc("COPA-eng", model)
evaluation["MCTest"] = eval_mc("MCTest-eng", model)
evaluation["MultiRC"] = eval_multirc("MultiRC-eng", model)
evaluation["SQUAD2"] = eval_squad2("SQUAD2-eng", model)

for dataset, evals in evaluation.items():
    print(f"*** {dataset} *** -> {evals}")




if checkpoint is not None:

    best_epoch = get_best_epoch(evaluation)
    print(f"Best epoch: {best_epoch + 1}")

    for dataset, evals in evaluation.items():
        print(f"*** {dataset} *** -> {evals[best_epoch]}")

    # plot_checkpoints(evaluation)
    # plot_squad2(evaluation)
    # plot_multirc(evaluation)




