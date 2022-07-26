import json
import jsonlines
import csv
import re
from os import listdir
from os.path import isfile, join
from collections import defaultdict
import string

fieldnames = ["input", "output", "type"]
kinds = ["train", "val", "test"]

def read_csv(path):
    lines = []
    with open(path, encoding='UTF8') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        l = [line for line in reader]
        lines += l
    return lines

def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

def remove_leading_and_trailing_punctuation_and_spaces(text):
    return text.strip().strip(string.punctuation).strip()

def boolq(dir_in, dir_out):
    for kind in kinds:
        writer = jsonlines.open(f"{dir_out}/BoolQ/{kind}.jsonl", mode="w")
        with jsonlines.open(f"{dir_in}/BoolQ/{kind}.jsonl") as reader:
            for line in reader:
                line_out = dict()
                question = line["question"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                if '?' not in question:
                    question = question + "?"
                paragraph = line["passage"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                line_out["input"] = question + " \\n " + paragraph
                if kind != "test": # test is without answers
                    answer = "da" if line["label"] else "ne"
                    line_out["output"] = answer
                writer.write(line_out)

def boolq_csv(dir_in, dir_out, dataset):
    d = f"{dir_in}/{dataset}"
    onlyfiles = [f for f in listdir(d) if isfile(join(d, f)) if f != "test.jsonl"]
    for file in onlyfiles:
        kind = file.split(".")[0]
        f = open(f"{dir_out}/{dataset}/{kind}.csv", 'w', encoding='UTF8', newline='')
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        with jsonlines.open(f"{dir_in}/{dataset}/{kind}.jsonl") as reader:
            for line in reader:
                line_out = dict()
                question = line["question"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                if '?' not in question:
                    question = question + "?"
                paragraph = line["passage"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                line_out["input"] = question + " \\n " + paragraph
                if kind != "test": # test is without answers
                    answer = "da" if line["label"] else "ne"
                    line_out["output"] = answer
                line_out["type"] = line["type"]
                # if '"' in line_out["input"]:
                #     print(line_out["input"])
                writer.writerow(line_out)

def multirc(dir_in, dir_out):
    for kind in kinds:
        writer = jsonlines.open(f"{dir_out}/MultiRC/{kind}.jsonl", mode="w")
        with jsonlines.open(f"{dir_in}/MultiRC/{kind}.jsonl") as reader:
            for line in reader:
                paragraph = line["passage"]["text"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                for q in line["passage"]["questions"]:
                    question = q["question"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                    if '?' not in question:
                        question = question + "?"
                    line_out = dict()
                    line_out["input"] = question + " \\n " + paragraph
                    if kind != "test":
                        for a in q["answers"]:
                            if a["label"] == 1:
                                line_out["output"] = a["text"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                                break
                    writer.write(line_out)

def multirc_csv(dir_in, dir_out, dataset):
    d = f"{dir_in}/{dataset}"
    onlyfiles = [f for f in listdir(d) if isfile(join(d, f)) if f != "test.jsonl"]
    for file in onlyfiles:
        kind = file.split(".")[0]
        f = open(f"{dir_out}/{dataset}/{kind}.csv", 'w', encoding='UTF8', newline='')
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        writer_answers = jsonlines.open(f"{dir_out}/answers/{dataset}-{kind}.jsonl", mode="w")
        with jsonlines.open(f"{dir_in}/{dataset}/{kind}.jsonl") as reader:
            for line in reader:
                paragraph = line["passage"]["text"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                for q in line["passage"]["questions"]:
                    question = q["question"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                    if '?' not in question:
                        question = question + "?"
                    line_out = dict()
                    line_out["input"] = question + " \\n " + paragraph
                    line_out["type"] = line["type"]
                    answers = {"answers": []}
                    if kind != "test":
                        for a in q["answers"]:
                            if a["label"] == 1:
                                answer = a["text"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                                answers["answers"].append(answer)

                        if answers["answers"] == []:
                            ans = " < Ni odgovora >"
                            answers["answers"].append(ans)

                        line_out["output"] = answers["answers"][0] # only take first correct answer into consideration for general sets (train, val, test_answered)
                        writer.writerow(line_out)
                        writer_answers.write(answers) # remember all correct answers

                    else:
                        writer.writerow(line_out)

def multirc_bin_csv(dir_in, dir_out):
    d = f"{dir_in}/MultiRC"
    onlyfiles = [f for f in listdir(d) if isfile(join(d, f)) if f != "test.jsonl"]
    for file in onlyfiles:
        kind = file.split(".")[0]
        f = open(f"{dir_out}/MultiRC-bin/{kind}.csv", 'w', encoding='UTF8', newline='')
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        with jsonlines.open(f"{dir_in}/MultiRC/{kind}.jsonl") as reader:
            for line in reader:
                paragraph = line["passage"]["text"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                for q in line["passage"]["questions"]:
                    question = q["question"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                    if '?' not in question:
                        question = question + "?"
                    line_out = dict()
                    line_out["input"] = question + " \\n " + paragraph
                    line_out["type"] = line["type"]
                    answers = {"answers": []}
                    if kind != "test":
                        for a in q["answers"]:
                            if a["label"] == 1:
                                answer = a["text"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                                answers["answers"].append(answer)

                        if answers["answers"] == []:
                            ans = " < Ni odgovora >"
                            answers["answers"].append(ans)

                        line_out["output"] = answers["answers"][0] # only take first correct answer into consideration for general sets (train, val, test_answered)
                        if line_out["output"].lower() in ["da", "ne"]:
                            writer.writerow(line_out)



# Remove repeating words from translated datasets such as Okno, okno ali Vitez, vitez
def remove_repeating_words(str):
    str2 = re.sub(r'[^\w\s]', '', str)
    str_split = [s.strip() for s in str2.split()]
    if len(str_split) == 2 and str_split[0].lower() == str_split[1].lower():
        return str_split[0]
    return str

def mctest_csv(dir_in, dir_out, dataset):
    d = f"{dir_in}/{dataset}"
    onlyfiles = [f for f in listdir(d) if isfile(join(d, f)) if f != "test.csv"]
    for file in onlyfiles:
        kind = file.split(".")[0]
        f = open(f"{dir_out}/{dataset}/{kind}.csv", 'w', encoding='UTF8', newline='')
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        with open(f"{dir_in}/{dataset}/{kind}.csv", encoding='UTF8') as f:
            reader = csv.reader(f)
            for line in reader:
                question = line[1].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ").replace("(angleščina)", "").replace("..", "")
                if '?' not in question:
                    question = question + "?"
                paragraph = line[0].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ").replace("(angleščina)", "").replace("..", "")
                input = question + " \\n"
                for letter, ans in zip(["A", "B", "C", "D"], line[2:-2]):
                    # if ans.endswith(" je"):
                    #     print(ans)
                    ans = remove_leading_and_trailing_punctuation_and_spaces(remove_repeating_words(ans.replace("(angleščina)", ""))).replace("  ", " ")
                    input += f" ({letter}) {ans}"
                input += " \\n " + paragraph
                output = remove_leading_and_trailing_punctuation_and_spaces(remove_repeating_words(line[-2].replace("(angleščina)", ""))).replace("  ", " ")
                line_out = dict()
                line_out["input"] = input
                line_out["output"] = output
                line_out["type"] = line[-1]

                writer.writerow(line_out)

def squad2_csv(dir_in, dir_out):
    d = f"{dir_in}/SQUAD2"
    onlyfiles = [f for f in listdir(d) if isfile(join(d, f)) if f != "test.csv"]
    for file in onlyfiles:
        kind = file.split(".")[0]
        f = open(f"{dir_out}/SQUAD2/{kind}.csv", 'w', encoding='UTF8', newline='')
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        with open(f"{dir_in}/SQUAD2/{kind}.csv", encoding='UTF8') as f:
            reader = csv.reader(f)
            for line in reader:

                paragraph = line[0].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ").replace("(angleščina)", "").replace("..", "")
                question = line[1].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ").replace("(angleščina)", "").replace("..", "")
                if '?' not in question:
                    question = question + "?"
                answer = remove_leading_and_trailing_punctuation_and_spaces(remove_repeating_words(line[2]).replace("(angleščina)", "")).replace("  ", " ") if line[2] != "< Ni odgovora >" else line[2]

                line_out = dict()
                line_out["input"] = question + " \\n " + paragraph
                line_out["output"] = answer
                line_out["type"] = line[-1]
                writer.writerow(line_out)

def squad_substring(dir_in):
    d = f"{dir_in}/SQUAD2"
    onlyfiles = [f for f in listdir(d) if isfile(join(d, f)) and f != "test.csv"]
    squad = []
    for file in onlyfiles:
        path = f"{d}/{file}"
        squad += read_csv(path)
    squad = [get_question_context_answer(line) for line in squad]
    squad = list(filter(lambda x: x[2] != "< Ni odgovora >", squad)) # only take into consideration those that have answers
    is_substring = [answer.lower() in context.lower() for _, context, answer in squad]
    print(f"Is answer substring of context? {is_substring.count(True)/len(is_substring):.2f}")
    # for is_sub, example in zip(is_substring, squad):
    #     if not is_sub:
    #         print(f"{example[2], example[1]}")


def get_question_context_answer(line):
    question = line[0].split("\\n")[0]
    paragraph = line[0].split("\\n")[1]
    answer = line[1]
    return question, paragraph, answer

def copa_csv(dir_in, dir_out):
    d = f"{dir_in}/COPA"
    onlyfiles = [f for f in listdir(d) if isfile(join(d, f)) if f != "test.jsonl"]
    for file in onlyfiles:
        kind = file.split(".")[0]
        f = open(f"{dir_out}/COPA/{kind}.csv", mode="w", encoding='UTF8', newline='')
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        with jsonlines.open(f"{dir_in}/COPA/{kind}.jsonl") as reader:
            for line in reader:
                line_out = dict()
                if line["question"] == "cause":
                    question = "Kaj je bil vzrok akcije?"
                elif line["question"] == "effect":
                    question = "Kaj se je zgodilo kot rezultat akcije?"
                else:
                    print(f"Ne ustreza vprašanjem {line['question']}")
                question = question.replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                paragraph = line["premise"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                choice1 = line["choice1"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                choice2 = line["choice2"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                input = f"{question} \\n (A) {choice1} (B) {choice2} \\n {paragraph}"
                line_out["input"] = input
                line_out["type"] = line["type"]

                if kind != "test": # test is without answers
                    output = choice1 if line["label"] == 0 else choice2
                    line_out["output"] = output

                writer.writerow(line_out)



def get_statistics(dir_in):
    datasets = ["BoolQ", "MCTest", "MultiRC", "SQUAD2", "COPA"]
    for dataset in datasets:
        counts = defaultdict(int)
        all = 0
        d = f"{dir_in}/{dataset}"
        onlyfiles = [f for f in listdir(d) if isfile(join(d, f))]
        for file in onlyfiles:
            kind = file.split(".")[0]
            with open(f"{dir_in}/{dataset}/{kind}.csv", encoding="utf8") as f:
                reader = csv.reader(f)
                next(reader) # don't count header
                number_of_lines = sum(1 for line in reader)
                counts[kind] = number_of_lines
                if kind != "test":
                    all += number_of_lines

        print(f"{dataset} train: {counts['train']/all:.2f}, val: {counts['val']/all:.2f}, test: {counts['test']/all:.2f}, test_answered: {counts['test_answered']/all:.2f}, absolute number of samples: {all}")

def get_absolute_statistics_table(dir_in):
    datasets = ["BoolQ", "MCTest", "MultiRC", "SQUAD2", "COPA"]
    for dataset in datasets:
        counts = defaultdict(int)
        all = 0
        d = f"{dir_in}/{dataset}"
        onlyfiles = [f for f in listdir(d) if isfile(join(d, f))]
        for file in onlyfiles:
            kind = file.split(".")[0]
            with open(f"{dir_in}/{dataset}/{kind}.csv", encoding="utf8") as f:
                reader = csv.reader(f)
                next(reader) # don't count header
                number_of_lines = sum(1 for line in reader)
                counts[kind] = number_of_lines
                if kind != "test":
                    all += number_of_lines

        print(f"{dataset} & {counts['train']} & {counts['val']} & {counts['test_answered']} \\\\")

def get_length(dir_in):
    datasets = ["BoolQ", "MCTest", "MultiRC", "SQUAD2", "COPA"]
    for dataset in datasets:
        d = f"{dir_in}/{dataset}"
        onlyfiles = [f for f in listdir(d) if isfile(join(d, f)) and f != "test.csv"]
        lines = []
        for file in onlyfiles:
            kind = file.split(".")[0]
            with open(f"{dir_in}/{dataset}/{kind}.csv", encoding="utf8") as f:
                reader = csv.reader(f)
                next(reader) # don't count header
                lines += [line for line in reader]
        questions = [line[0].split("\\n")[0] for line in lines]
        paragraphs = [line[0].split("\\n")[1] for line in lines]
        answers = [line[1] for line in lines]

        for name, text in zip(["question", "paragraph", "answer"],[questions, paragraphs, answers]):
            lengths = list(map(lambda x: len(x), text))
            print(f"{dataset}: {name} - average length {float(sum(lengths)/len(lengths)):.2f}")

def find_human_translation_if_exists(d, ht_data, mt_data):
    assert len(ht_data) == len(mt_data)

    for ind, (ht, mt) in enumerate(zip(ht_data, mt_data)):
        if mt["question"] == d["question"] and mt["context"] == d["context"]:
            return ht
    return None

def squad2_project_csv(dir_in, dir_out):
    kinds = ["train", "val"]
    with open(f"{dir_in}/SQUAD2-project/ht.json", encoding='UTF8') as f:
        ht_data = json.load(f)["data"]
    with open(f"{dir_in}/SQUAD2-project/mt.json", encoding='UTF8') as f:
        mt_data = json.load(f)["data"]

    for kind in kinds:
        # split train set into train and validation and use val set as test set
        if kind == "train":
            f_val = open(f"{dir_out}/SQUAD2-project/val.csv", mode="w", encoding='UTF8', newline='')
            writer_val = csv.DictWriter(f_val, fieldnames=fieldnames)
            writer_val.writeheader()

            f = open(f"{dir_out}/SQUAD2-project/train.csv", mode="w", encoding='UTF8', newline='')
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            writer_val_results = jsonlines.open(f"{dir_out}/answers/SQUAD2-project-val.jsonl", mode="w")
            writer_results = jsonlines.open(f"{dir_out}/answers/SQUAD2-project-train.jsonl", mode="w")


        elif kind == "val":
            f = open(f"{dir_out}/SQUAD2-project/test_answered.csv", mode="w", encoding='UTF8', newline='')
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            writer_results = jsonlines.open(f"{dir_out}/answers/SQUAD2-project-test_answered.jsonl", mode="w")

        else:
            raise "Non anticipated dataset versions"

        with open(f"{dir_in}/SQUAD2-project/{kind}.json", encoding='UTF8') as f:
            data = json.load(f)["data"]
            for ind, d in enumerate(data):
                ht = find_human_translation_if_exists(d, ht_data, mt_data)
                type = "MT"
                if ht is not None:
                    d = ht
                    type = "HT"
                context = d["context"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ").replace(" ", "")
                question = d["question"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ").replace(" ", "")


                line_out = dict()
                input = f"{question} \\n {context}"

                # get all unique answers and filter out empty ones
                answers = list(filter(lambda y: y != "",
                                    set(map(lambda x: remove_leading_and_trailing_punctuation_and_spaces(
                                                        x.replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ").replace(" ", "")),
                                                        d["answers"]["text"]))))
                if len(answers) == 0:
                    answers = ["< Ni odgovora >"]

                output = answers[0]
                line_out["input"] = input
                line_out["output"] = output
                line_out["type"] = type
                if kind == "train" and ind >= 115000:
                    writer_val.writerow(line_out)
                    writer_val_results.write({"answers": answers})
                else:
                    writer.writerow(line_out)
                    writer_results.write({"answers": answers})

def squad2_project_ht_mt(dir_in, dir_out):
    types = ["ht", "mt"]
    for type in types:
        with open(f"{dir_in}/SQUAD2-project/{type}.json", encoding='UTF8') as f:
            data = json.load(f)["data"]


        f = open(f"{dir_out}/SQUAD2-project-{type}/test_answered.csv", mode="w", encoding='UTF8', newline='')
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer_results = jsonlines.open(f"{dir_out}/answers/SQUAD2-project-{type}-test_answered.jsonl", mode="w")

        for ind, d in enumerate(data):

            context = d["context"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ").replace(" ", "")
            question = d["question"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ").replace(" ", "")

            line_out = dict()
            input = f"{question} \\n {context}"

            # get all unique answers and filter out empty ones
            answers = list(filter(lambda y: y != "",
                                set(map(lambda x: remove_leading_and_trailing_punctuation_and_spaces(
                                                    x.replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ").replace(" ", "")),
                                                    d["answers"]["text"]))))
            if len(answers) == 0:
                answers = ["< Ni odgovora >"]

            output = answers[0]
            line_out["input"] = input
            line_out["output"] = output
            line_out["type"] = type.upper()

            writer.writerow(line_out)
            writer_results.write({"answers": answers})




dir_in = "../../../Magistrska/Datasets/ALL"
dir_out = "encoded"
# boolq_csv(dir_in, dir_out, "BoolQ")
multirc_csv(dir_in, dir_out, "MultiRC")
# multirc_bin_csv(dir_in, dir_out)
# mctest_csv(dir_in, dir_out, "MCTest")
# squad2_csv(dir_in, dir_out)
# squad2_project_csv(dir_in, dir_out)
# copa_csv(dir_in, dir_out)

# squad_substring(dir_out)
# get_length(dir_out)
# get_statistics(dir_out)
# get_absolute_statistics_table(dir_out)

# squad2_project_ht_mt(dir_in, dir_out)
mctest_csv(dir_in, dir_out, "MCTest-deepl")
mctest_csv(dir_in, dir_out, "MCTest-mt")
# boolq_csv(dir_in, dir_out, "BoolQ-ht")
# boolq_csv(dir_in, dir_out, "BoolQ-mt")
# multirc_csv(dir_in, dir_out, "MultiRC-ht")
# multirc_csv(dir_in, dir_out, "MultiRC-mt")
