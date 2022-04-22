import json
import jsonlines
import re
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def get_statistics(dir, datasets, kinds):
    for dataset in datasets:
        for kind in kinds:
            with open(f"{dir}/HT/{dataset}/{kind}.jsonl", encoding="utf-8") as f:
                ht_lines = len(f.readlines())
            if dataset != "COPA" or kind != "test_answered": #COPA doesn't have answers in MT
                with open(f"{dir}/MT/{dataset}/{kind}.jsonl", encoding="utf-8") as f:
                    mt_lines = len(f.readlines())
            with open(f"{dir}/ALL/{dataset}/{kind}.jsonl", encoding="utf-8") as f:
                all_lines = len(f.readlines())
            print(f"{dataset} {kind} human translated {ht_lines}, machine translated {mt_lines}, all {all_lines}")

def find_property_with_value(array, property, value):
    for item in array:
        if item[property] == value:
            return True
    return False

# This method was needed to correctly match questions / answers where some special characters were interpreted differently
def remove_special_chars(str):
    return re.sub('[^A-Za-z0-9]+', '', str)

def merge(datasets, kinds, dir):
    for dataset in datasets:
        for kind in kinds:
            data_ht = []
            with jsonlines.open(f"{dir}/HT/{dataset}/{kind}.jsonl") as reader:
                for line in reader:
                    data_ht.append(line)
            data_mt = []
            # Only add those machine translated lines that are not in human translated lines
            if dataset != "COPA" or kind != "test_answered": # COPA doens't have answers in MT
                with jsonlines.open(f"{dir}/MT/{dataset}/{kind}.jsonl") as reader:
                    for line in reader:
                        if not find_property_with_value(data_ht, "idx", line["idx"]):
                            data_mt.append(line)
            data = data_ht + data_mt
            with jsonlines.open(f"{dir}/ALL/{dataset}/{kind}.jsonl", mode="w") as writer:
                for line in data:
                    writer.write(line)


def add_answers_multirc_test(dir):
    f1 = open(f"{dir}/MT/MultiRC/testprep/test1.json") # previously test_1_83-fixedIds.json
    data1 = json.load(f1)["data"]

    f2 = open(f"{dir}/MT/MultiRC/testprep/test2.json") # previous test_2_83-fixedIds.json
    data2 = json.load(f2)["data"]

    lines = []

    with jsonlines.open(f"{dir}/MT/MultiRC/testprep/test.jsonl") as reader: # English test.jsonl
        for ind, line in enumerate(reader):
            questions = line["passage"]["questions"]
            for question in questions:
                for data in [data1, data2]:
                    for d in data:
                        for question2 in d["paragraph"]["questions"]:
                            q = question2["question"]
                            if remove_special_chars(q) == remove_special_chars(question["question"]):
                                for answer in question["answers"]:
                                    for ans in question2["answers"]:
                                        if remove_special_chars(answer["text"]) == remove_special_chars(ans["text"]):
                                            answer["label"] = 1 if ans["isAnswer"] else 0

            lines.append(line)

    writer = jsonlines.open(f"{dir}/MT/MultiRC/test_answered.jsonl", mode="w") # Slovene test_answered.jsonl
    with jsonlines.open(f"{dir}/MT/MultiRC/test.jsonl") as reader: # Slovene test.jsonl
        for ind, l in enumerate(reader):
            for ind2, question in enumerate(l["passage"]["questions"]):
                for ind3, ans in enumerate(question["answers"]):
                    try:
                        ans["label"] = lines[ind]["passage"]["questions"][ind2]["answers"][ind3]["label"]
                    except:
                        # This happens in case some answers were not matched and miss label
                        print(remove_special_chars(lines[ind]["passage"]["questions"][ind2]["question"]))
                        print(remove_special_chars(lines[ind]["passage"]["questions"][ind2]["answers"][ind3]["text"]))
            writer.write(l)



def add_answers_boolq_test(dir):
    writer = jsonlines.open(f"{dir}/MT/BoolQ/test_answered.jsonl", mode="w")
    with jsonlines.open(f"{dir}/MT/BoolQ/test.jsonl") as reader1:
        with jsonlines.open(f"{dir}/MT/BoolQ/test_answers.jsonl") as reader2:
            data = [line for line in reader1]
            answers = [line for line in reader2]
            for line, answer in zip(data, answers):
                if line["idx"] == answer["idx"]:
                    line["label"] = answer["label"]
                    writer.write(line)
                else:
                    raise ValueError



# dir = "C:/Users/Katja/Documents/FRI/Magistrska/Datasets"
datasets = ["BoolQ", "COPA", "MultiRC"]
kinds = ["train", "val", "test", "test_answered"]
dir = "../../../Magistrska/Datasets"

merge(datasets, kinds, dir)

# add_answers_boolq_test(dir) # ni več teh datotek
# add_answers_multirc_test(dir)

get_statistics(dir, datasets, kinds)