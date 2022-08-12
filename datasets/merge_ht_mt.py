import json
import jsonlines
import re
from difflib import SequenceMatcher
import csv

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

def get_statistics_table(dir, datasets, kinds):
    for kind in kinds:
        print(f"{kind} ", end="")
        for dataset in datasets:
            with open(f"{dir}/HT/{dataset}/{kind}.jsonl", encoding="utf-8") as f:
                ht_lines = len(f.readlines())
            if dataset != "COPA" or kind != "test_answered": #COPA doesn't have answers in MT
                with open(f"{dir}/MT/{dataset}/{kind}.jsonl", encoding="utf-8") as f:
                    mt_lines = len(f.readlines())
            with open(f"{dir}/ALL/{dataset}/{kind}.jsonl", encoding="utf-8") as f:
                all_lines = len(f.readlines())
            print(f"{ht_lines/all_lines:.2f} & {(mt_lines-ht_lines)/all_lines:.2f} & ", end="") # wrong for copa machine translated because it doesn't have test_answered file
        print("\\\\")

def find_property_with_value(array, property, value):
    for item in array:
        if item[property] == value:
            return True
    return False

def find_id_csv(array, value):
    for item in array:
        if item[0] == value:
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
                    line["type"] = "HT"
                    data_ht.append(line)
            data_mt = []
            # Only add those machine translated lines that are not in human translated lines
            if dataset != "COPA" or kind != "test_answered": # COPA doens't have answers in MT
                with jsonlines.open(f"{dir}/MT/{dataset}/{kind}.jsonl") as reader:
                    for line in reader:
                        if not find_property_with_value(data_ht, "idx", line["idx"]):
                            line["type"] = "MT"
                            data_mt.append(line)
            data = data_ht + data_mt
            with jsonlines.open(f"{dir}/ALL/{dataset}/{kind}.jsonl", mode="w") as writer:
                for line in data:
                    writer.write(line)

# create datasets with same examples for human and machine translation
def create_ht_mt(datasets, kinds, dir):
    for dataset in datasets:
        for kind in kinds:
            data_ht = []
            with jsonlines.open(f"{dir}/HT/{dataset}/{kind}.jsonl") as reader:
                for line in reader:
                    line["type"] = "HT"
                    data_ht.append(line)
            data_mt = []
            # Only add those machine translated lines that are in human translated lines
            with jsonlines.open(f"{dir}/MT/{dataset}/{kind}.jsonl") as reader:
                for line in reader:
                    if find_property_with_value(data_ht, "idx", line["idx"]):
                        line["type"] = "MT"
                        data_mt.append(line)
            with jsonlines.open(f"{dir}/ALL/{dataset}-ht/{kind}.jsonl", mode="w") as writer:
                for line in data_ht:
                    writer.write(line)
            with jsonlines.open(f"{dir}/ALL/{dataset}-mt/{kind}.jsonl", mode="w") as writer:
                for line in data_mt:
                    writer.write(line)

def merge_mctest_deepl(dir_in):
    data_deepl = []
    with open(f"{dir_in}/MCTest-deepl.csv", encoding='UTF8') as f:
        csv_file = csv.reader(f)
        for line in csv_file:
            line.append("DEEPL")
            data_deepl.append(line)
    data_mt = []
    # Only add those machine translated lines that are not in deepl translated lines
    with open(f"{dir_in}/MCTest-sl.csv", encoding='UTF8') as f:
        csv_file = csv.reader(f)
        for line in csv_file:
            if not find_id_csv(data_deepl, line[0]):
                line.append("MT")
                data_mt.append(line)
    data = data_deepl + data_mt
    data.sort(key=lambda k: (k[0][3:5], int(k[0][:3]), int(k[0][5:]))) # sort it the same as previously (kind (train, test, dev), 150/160, id)
    f_out = open(f"{dir_in}/MCTest-merged.csv", 'w', encoding='UTF8', newline='')
    writerc = csv.writer(f_out)
    writerc.writerows(data)

# create the mctest datasets with same examples with deepl translation and machine translation
def create_deepl_mt(dir_in):
    data_deepl = []
    with open(f"{dir_in}/MCTest-deepl.csv", encoding='UTF8') as f:
        csv_file = csv.reader(f)
        for line in csv_file:
            line.append("DEEPL")
            data_deepl.append(line)
    data_mt = []
    # Only add those machine translated lines that are  in deepl translated lines
    with open(f"{dir_in}/MCTest-sl.csv", encoding='UTF8') as f:
        csv_file = csv.reader(f)
        for line in csv_file:
            if find_id_csv(data_deepl, line[0]):
                line.append("MT")
                data_mt.append(line)

    # Only save machine translated examples, data_deepl hasn't changed and is still available in MCTest-deepl.csv
    f_out = open(f"{dir_in}/MCTest-mt.csv", 'w', encoding='UTF8', newline='')
    data_mt.sort(key=lambda k: (k[0][3:5], int(k[0][:3]), int(k[0][5:])))  # sort it the same as previously (kind (train, test, dev), 150/160, id)
    writerc = csv.writer(f_out)
    writerc.writerows(data_mt)

def merge_squad_deepl(dir_in):
    kinds = ["train", "val"]

    for kind in kinds:
        data_deepl = []
        with open(f"{dir_in}/SQUAD-deepl-{kind}-qa.csv", encoding='UTF8') as f:
            csv_file = csv.reader(f)
            for line in csv_file:
                if "<" in line[-1]:
                    line[-1] = "< Ni odgovora >"
                line.append("DEEPL")
                data_deepl.append(line)
        data_mt = []
        with open(f"{dir_in}/Squad-{kind}-qa-sl.csv", encoding='UTF8') as f:
            csv_file = csv.reader(f)
            for line in csv_file:
                if "<" in line[-1]:
                    line[-1] = "< Ni odgovora >"
                line.append("MT")
                data_mt.append(line)
        if kind == "train":
            data = data_deepl[:1000] + data_mt[1000:-700] + data_deepl[-700:]
        elif kind == "val":
            data = data_deepl[:700] + data_mt[700:]
        f_out = open(f"{dir_in}/Squad-{kind}-qa-merged.csv", 'w', encoding='UTF8', newline='')
        writerc = csv.writer(f_out)
        writerc.writerows(data)

# Find matching answer by id
def get_answer(id, answers):
    for answer in answers:
        if id == answer["idx"]:
            return answer
    raise "Cannot find matching ids"

def add_answers_multirc_test(dir):
    f1 = open(f"{dir}/MT/MultiRC/testprep/test1.json")  # previously test_1_83-fixedIds.json
    data1 = json.load(f1)["data"]

    f2 = open(f"{dir}/MT/MultiRC/testprep/test2.json")  # previous test_2_83-fixedIds.json
    data2 = json.load(f2)["data"]

    lines = []

    with jsonlines.open(f"{dir}/MT/MultiRC/testprep/test.jsonl") as reader:  # English test.jsonl
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

    # Machine translated
    writer = jsonlines.open(f"{dir}/MT/MultiRC/test_answered.jsonl", mode="w")  # Slovene test_answered.jsonl
    with jsonlines.open(f"{dir}/MT/MultiRC/test.jsonl") as reader:  # Slovene test.jsonl
        for ind, l in enumerate(reader):
            for ind2, question in enumerate(l["passage"]["questions"]):
                for ind3, ans in enumerate(question["answers"]):
                    # try:
                    answers = lines[ind]["passage"]["questions"][ind2]["answers"]
                    ans["label"] = get_answer(ans["idx"], answers)["label"]
                # except:
                #     # This happens in case some answers were not matched and miss label
                #     print(remove_special_chars(lines[ind]["passage"]["questions"][ind2]["question"]))
                #     print(remove_special_chars(lines[ind]["passage"]["questions"][ind2]["answers"][ind3]["text"]))
            writer.write(l)

    # Human translated
    writer = jsonlines.open(f"{dir}/HT/MultiRC/test_answered.jsonl", mode="w")  # Slovene test_answered.jsonl
    with jsonlines.open(f"{dir}/HT/MultiRC/test.jsonl") as reader:  # Slovene test.jsonl
        for ind, l in enumerate(reader):
            for ind2, question in enumerate(l["passage"]["questions"]):
                for ind3, ans in enumerate(question["answers"]):
                    # try:
                    answers = lines[ind]["passage"]["questions"][ind2]["answers"]
                    ans["label"] = get_answer(ans["idx"], answers)["label"]
                # except:
                #     # This happens in case some answers were not matched and miss label
                #     print(remove_special_chars(lines[ind]["passage"]["questions"][ind2]["question"]))
                #     print(remove_special_chars(lines[ind]["passage"]["questions"][ind2]["answers"][ind3]["text"]))
            writer.write(l)

def check_answers_multirc(dir):
    with jsonlines.open(f"{dir}/HT/MultiRC/test_answered.jsonl") as reader_ht:
        with jsonlines.open(f"{dir}/MT/MultiRC/test_answered.jsonl") as reader_mt:
            with jsonlines.open(f"{dir}/English/MultiRC/test_answered.jsonl") as reader_eng:

                for line_ht, line_mt, line_eng in zip(reader_ht, reader_mt, reader_eng):
                    assert line_ht["idx"] == line_mt["idx"] == line_eng["idx"], "Human and machine translation passages do not match"
                    questions_ht = line_ht["passage"]["questions"]
                    questions_mt = line_mt["passage"]["questions"]
                    questions_eng = line_eng["passage"]["questions"]
                    for question_ht, question_mt, question_eng in zip(questions_ht, questions_mt, questions_eng):
                        assert question_ht["idx"] == question_mt["idx"] == question_eng["idx"], "Human and machine translations questions do not match"
                        answers_ht = question_ht["answers"]
                        answers_mt = question_mt["answers"]
                        answers_eng = question_eng["answers"]
                        for answer_ht, answer_mt in zip(answers_ht, answers_mt):
                            answer_eng = get_answer(answer_ht["idx"], answers_eng)
                            assert answer_ht["idx"] == answer_mt["idx"] == answer_eng["idx"], "Human and machine translations answers do not match"
                            assert answer_ht["label"] == answer_mt["label"] == answer_eng["label"], "Human and machine translations labels do not match"
                            # if answer_ht["label"] != answer_eng["label"]:
                            #     print(question_ht["idx"])
                            #     print(answer_ht)
                            #     print(answer_mt)
                            #     print(answer_eng)
                            #     print()

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
datasets = ["BoolQ", "MultiRC", "COPA"]
kinds = ["train", "val",  "test_answered"]
dir = "../../../Magistrska/Datasets"

# merge(datasets, kinds, dir)
# merge_mctest_deepl("../../../Magistrska/Datasets/MT/translationprep")
# merge_squad_deepl("../../../Magistrska/Datasets/MT/translationprep")

# add_answers_boolq_test(dir) # ni več teh datotek
# add_answers_multirc_test(dir)
check_answers_multirc(dir)

# create_ht_mt(["BoolQ", "MultiRC"], kinds, dir)
# create_deepl_mt("../../../Magistrska/Datasets/MT/translationprep")

get_statistics(dir, datasets, kinds)
get_statistics_table(dir, datasets, kinds)