import json
import jsonlines
import csv
import re
from os import listdir
from os.path import isfile, join
from collections import defaultdict

import pandas as pd

from my_utils import read_csv
import string
from typing import List
import jsonlines

fieldnames = ["input", "output", "type"]
kinds = ["train", "val", "test"]

def squad2_csv(dir_in, dir_out):
    kinds = ["train", "val"]

    for kind in kinds:
        # split train set into train and validation and use val set as test set
        if kind == "train":
            f_val = open(f"{dir_out}/SQUAD2-eng/val.csv", mode="w", encoding='UTF8', newline='')
            writer_val = csv.DictWriter(f_val, fieldnames=fieldnames)
            writer_val.writeheader()

            f = open(f"{dir_out}/SQUAD2-eng/train.csv", mode="w", encoding='UTF8', newline='')
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            writer_val_results = jsonlines.open(f"{dir_out}/answers/SQUAD2-eng-val.jsonl", mode="w")
            writer_results = jsonlines.open(f"{dir_out}/answers/SQUAD2-eng-train.jsonl", mode="w")


        elif kind == "val":
            f = open(f"{dir_out}/SQUAD2-eng/test_answered.csv", mode="w", encoding='UTF8', newline='')
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            writer_results = jsonlines.open(f"{dir_out}/answers/SQUAD2-eng-test_answered.jsonl", mode="w")

        else:
            raise "Non anticipated dataset versions"

        with open(f"{dir_in}/SQUAD2/{kind}.json", encoding='UTF8') as f:
            data = json.load(f)["data"]
            for ind, d in enumerate(data):
                type = "ORIG"
                context = d["context"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ").replace(" ", "")
                question = d["question"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ").replace(" ", "")


                line_out = dict()
                input = f"{question} \\n {context}"

                # get all unique answers and filter out empty ones
                answers = list(filter(lambda y: y != "",
                                    set(map(lambda x: x.replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ").replace(" ", ""),
                                                        d["answers"]["text"]))))
                if len(answers) == 0:
                    answers = ["<No answer>"]

                output = answers[0]
                line_out["input"] = input
                line_out["output"] = output
                line_out["type"] = type
                if kind == "train" and ind >= 119569:
                    writer_val.writerow(line_out)
                    writer_val_results.write({"answers": answers})
                else:
                    writer.writerow(line_out)
                    writer_results.write({"answers": answers})

def mctest_read_answers(dir_in, kind):
    sizes = [160, 500]
    all_answers = []
    for size in sizes:
        path = f"{dir_in}/MCTest/mc{size}.{kind}.ans"
        with open(path.replace(".tsv", ".ans")) as f:
            for l in f.readlines():
                all_answers.extend(l.replace("\n", "").split("\t"))
    return all_answers

def get_answer(ans):
    if ans == "A":
        return 0
    elif ans == "B":
        return 1
    elif ans == "C":
        return 2
    elif ans == "D":
        return 3

def mctest_csv(dir_in, dir_out):
    kinds = ["train", "dev", "test"]
    sizes = [160,500]
    for kind in kinds:
        all_candidates = []
        all_inputs = []
        for size in sizes:
            with open(f"{dir_in}/MCTest/mc{size}.{kind}.tsv") as f:
                for line in f.readlines():
                    line_split = line.replace("\n", "").replace("\\newline", " ").split("\t")
                    paragraph = line_split[2]

                    def get_question_and_candidates(split_row: List[str]):
                        question = split_row[0].replace("one: ", "").replace("multiple: ", "")
                        candidates = split_row[1:5]
                        all_candidates.append(candidates)
                        candidates = " ".join([f"({chr(ord('A') + i)}) {x}" for i, x in enumerate(candidates)])
                        all_inputs.append(f"{question} \\n {candidates} \\n {paragraph}")

                    get_question_and_candidates(line_split[3:8])
                    get_question_and_candidates(line_split[8:13])
                    get_question_and_candidates(line_split[13:18])
                    get_question_and_candidates(line_split[18:23])

        all_answers = mctest_read_answers(dir_in, kind)

        if kind == "test":
            kind = "test_answered"
        if kind == "dev":
            kind = "val"
        f = open(f"{dir_out}/MCTest-eng/{kind}.csv", 'w', encoding='UTF8', newline='')
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        type = "ORIG"
        for input, ans, candidates in zip(all_inputs, all_answers, all_candidates):
            line_out = dict()
            output = candidates[get_answer(ans)]
            line_out["input"] = input
            line_out["output"] = output
            line_out["type"] = type
            writer.writerow(line_out)

def multirc_csv(dir_in, dir_out):
    d = f"{dir_in}/MultiRC"
    onlyfiles = [f for f in listdir(d) if isfile(join(d, f)) if f != "test.jsonl"]
    for file in onlyfiles:
        kind = file.split(".")[0]
        f = open(f"{dir_out}/MultiRC-eng/{kind}.csv", 'w', encoding='UTF8', newline='')
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        writer_answers = jsonlines.open(f"{dir_out}/answers/MultiRC-eng-{kind}.jsonl", mode="w")
        with jsonlines.open(f"{dir_in}/MultiRC/{kind}.jsonl") as reader:
            for line in reader:
                paragraph = line["passage"]["text"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                for q in line["passage"]["questions"]:
                    question = q["question"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                    if '?' not in question:
                        question = question + "?"
                    line_out = dict()
                    line_out["input"] = question + " \\n " + paragraph
                    line_out["type"] = "ORIG"
                    answers = {"answers": []}
                    if kind != "test":
                        for a in q["answers"]:
                            if a["label"] == 1:
                                answer = a["text"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                                answers["answers"].append(answer)

                        if answers["answers"] == []:
                            ans = " <No answer>"
                            answers["answers"].append(ans)

                        line_out["output"] = answers["answers"][0] # only take first correct answer into consideration for general sets (train, val, test_answered)
                        writer.writerow(line_out)
                        writer_answers.write(answers) # remember all correct answers

                    else:
                        writer.writerow(line_out)


def boolq_csv(dir_in, dir_out):
    d = f"{dir_in}/BoolQ"
    onlyfiles = [f for f in listdir(d) if isfile(join(d, f)) if f != "test.jsonl"]
    for file in onlyfiles:
        kind = file.split(".")[0]
        f = open(f"{dir_out}/BoolQ-eng/{kind}.csv", 'w', encoding='UTF8', newline='')
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        with jsonlines.open(f"{dir_in}/BoolQ/{kind}.jsonl") as reader:
            for line in reader:
                line_out = dict()
                question = line["question"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                if '?' not in question:
                    question = question + "?"
                paragraph = line["passage"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                line_out["input"] = question + " \\n " + paragraph
                answer = "yes" if line["label"] else "no"
                line_out["output"] = answer
                line_out["type"] = "ORIG"
                # if '"' in line_out["input"]:
                #     print(line_out["input"])
                writer.writerow(line_out)

def copa_csv(dir_in, dir_out):
    d = f"{dir_in}/COPA"
    onlyfiles = [f for f in listdir(d) if isfile(join(d, f)) if f != "test.jsonl"]
    for file in onlyfiles:
        kind = file.split(".")[0]
        f = open(f"{dir_out}/COPA-eng/{kind}.csv", mode="w", encoding='UTF8', newline='')
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        with jsonlines.open(f"{dir_in}/COPA/{kind}.jsonl") as reader:
            for line in reader:
                line_out = dict()
                if line["question"] == "cause":
                    question = "What was the cause of action?"
                elif line["question"] == "effect":
                    question = "What was the result of action?"
                else:
                    print(f"Ne ustreza vprašanjem {line['question']}")
                question = question.replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                paragraph = line["premise"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                choice1 = line["choice1"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                choice2 = line["choice2"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                input = f"{question} \\n (A) {choice1} (B) {choice2} \\n {paragraph}"
                line_out["input"] = input
                line_out["type"] = "ORIG"

                if kind != "test": # test is without answers
                    output = choice1 if line["label"] == 0 else choice2
                    line_out["output"] = output

                writer.writerow(line_out)

def add_answers_to_test(dir_eng, dir_sl, dataset):
    with jsonlines.open(f"{dir_eng}/{dataset}/test.jsonl") as reader:
        lines_eng = [line for line in reader]
    with jsonlines.open(f"{dir_sl}/{dataset}/test_answered.jsonl") as reader:
        lines_sl = [line for line in reader]

    assert len(lines_eng) == len(lines_sl)

    lines_eng_answered = []
    for eng, sl in zip(lines_eng, lines_sl):
        eng["label"] = sl["label"]
        lines_eng_answered.append(eng)

    writer = jsonlines.open(f"{dir_eng}/{dataset}/test_answered.jsonl",mode="w")
    writer.write_all(lines_eng_answered)


def convert_tsv_to_csv(dir_in, dir_out, dataset):
    d = f"{dir_in}/{dataset}"
    onlyfiles = [f for f in listdir(d) if isfile(join(d, f))]
    for file in onlyfiles:
        data = pd.read_table(f"{d}/{file}")
        # if file == "test.csv":
        #     data["output"] = "" # test files do not have answers
        data["type"] = "ORIG"
        output_path = f"{dir_out}/{dataset}-eng/{file.split('.')[0]}.csv"
        data.to_csv(output_path, index=False, header=fieldnames)


dir_in = "../../../Magistrska/Datasets/English"
dir_out = "encoded"

add_answers_to_test(dir_in, "../../../Magistrska/Datasets/ALL", "BoolQ")
add_answers_to_test(dir_in, "../../../Magistrska/Datasets/ALL", "COPA")

mctest_csv(dir_in, dir_out)
squad2_csv(dir_in, dir_out)
multirc_csv(dir_in, dir_out)
boolq_csv(dir_in, dir_out)
copa_csv(dir_in, dir_out)


convert_tsv_to_csv(dir_in, dir_out, "ARC-easy")
convert_tsv_to_csv(dir_in, dir_out, "ARC-hard")
convert_tsv_to_csv(dir_in, dir_out, "NarrativeQA")
convert_tsv_to_csv(dir_in, dir_out, "OBQA")
convert_tsv_to_csv(dir_in, dir_out, "RACE")


