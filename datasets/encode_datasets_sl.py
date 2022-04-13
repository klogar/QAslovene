import json
import jsonlines
import csv
import re

fieldnames = ["input", "output"]
kinds = ["train", "val", "test"]

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
                line_out["input"] = question + " \n " + paragraph
                if kind != "test": # test is without answers
                    answer = "da" if line["label"] else "ne"
                    line_out["output"] = answer
                writer.write(line_out)

def boolq_csv(dir_in, dir_out):
    for kind in kinds:
        f = open(f"{dir_out}/BoolQ/{kind}.csv", 'w', encoding='UTF8', newline='')
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        with jsonlines.open(f"{dir_in}/BoolQ/{kind}.jsonl") as reader:
            for line in reader:
                line_out = dict()
                question = line["question"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                if '?' not in question:
                    question = question + "?"
                paragraph = line["passage"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                line_out["input"] = question + " \n " + paragraph
                if kind != "test": # test is without answers
                    answer = "da" if line["label"] else "ne"
                    line_out["output"] = answer
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
                    line_out["input"] = question + " \n " + paragraph
                    if kind != "test":
                        for a in q["answers"]:
                            if a["label"] == 1:
                                line_out["output"] = a["text"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                                break
                    writer.write(line_out)

def multirc_csv(dir_in, dir_out):
    for kind in kinds:
        f = open(f"{dir_out}/MultiRC/{kind}.csv", 'w', encoding='UTF8', newline='')
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
                    line_out["input"] = question + " \n " + paragraph
                    if kind != "test":
                        for a in q["answers"]:
                            if a["label"] == 1:
                                line_out["output"] = a["text"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                                # if '"' in line_out["output"]:
                                #     print(line_out["output"])

                                writer.writerow(line_out)
                    else:
                        writer.writerow(line_out)
                    # if '"' in line_out["input"]:
                    #     print(line_out["input"])

# Remove repeating words from translated datasets such as Okno, okno ali Vitez, vitez
def remove_repeating_words(str):
    str2 = re.sub(r'[^\w\s]', '', str)
    str_split = [s.strip() for s in str2.split()]
    if len(str_split) == 2 and str_split[0].lower() == str_split[1].lower():
        return str_split[0]
    return str

def mctest_csv(dir_in, dir_out):
    for kind in kinds:
        f = open(f"{dir_out}/MCTest/{kind}.csv", 'w', encoding='UTF8', newline='')
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        with open(f"{dir_in}/MCTest/{kind}.csv", encoding='UTF8') as f:
            reader = csv.reader(f)
            for line in reader:
                question = line[1].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ").replace("(angleščina)", "").replace("..", "")
                if '?' not in question:
                    question = question + "?"
                paragraph = line[0].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ").replace("(angleščina)", "").replace("..", "")
                input = question + " \n"
                for letter, ans in zip(["A", "B", "C", "D"], line[2:-1]):
                    ans = ans[:-1] if ans.endswith(".") else ans #remove dot
                    ans = ans.replace("(angleščina)", "").replace("..", "")
                    ans = remove_repeating_words(ans)
                    input += f" ({letter}) {ans}"
                input += " \n " + paragraph
                output = remove_repeating_words(line[-1])
                line_out = dict()
                line_out["input"] = input
                line_out["output"] = output

                writer.writerow(line_out)

def squad2_csv(dir_in, dir_out):
    for kind in kinds:
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
                answer = remove_repeating_words(line[2]).replace("(angleščina)", "").replace("..", "")

                line_out = dict()
                line_out["input"] = question + " \n " + paragraph
                line_out["output"] = answer
                writer.writerow(line_out)

def copa_csv(dir_in, dir_out):
    for kind in kinds:
        f = open(f"{dir_out}/COPA/{kind}.csv", mode="w", encoding='UTF8', newline='')
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        with jsonlines.open(f"{dir_in}/COPA/{kind}.jsonl") as reader:
            for line in reader:
                line_out = dict()
                if line["question"] == "cause":
                    question = "Kaj se je zgodilo kot rezultat akcije?"
                elif line["question"] == "effect":
                    question = "Kaj je bil vzrok akcije?"
                else:
                    print(f"Ne ustreza vprašanjem {line['question']}")
                question = question.replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                paragraph = line["premise"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                choice1 = line["choice1"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                choice2 = line["choice2"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                input = f"{question} \n (A) {choice1} (B) {choice2} \n {paragraph}"
                line_out["input"] = input
                if kind != "test": # test is without answers
                    output = choice1 if line["label"] == 0 else choice2
                    line_out["output"] = output

                writer.writerow(line_out)



def get_statistics(dir_in):
    datasets = ["BoolQ", "MCTest", "MultiRC", "SQUAD2", "COPA"]
    kinds = ["train", "val", "test"]
    for dataset in datasets:
        counts = dict()
        all = 0
        for kind in kinds:
            with open(f"{dir_in}/{dataset}/{kind}.csv", encoding="utf8") as f:
                reader = csv.reader(f)
                number_of_lines = sum(1 for line in reader)
                counts[kind] = number_of_lines
                all += number_of_lines

        print(f"{dataset} train: {counts['train']/all:.2f}, val: {counts['val']/all:.2f}, test: {counts['test']/all:.2f}, absolute number of samples: {all}")





dir_in = "../../../Magistrska/Datasets/ALL"
dir_out = "encoded"
boolq_csv(dir_in, dir_out)
multirc_csv(dir_in, dir_out)
mctest_csv(dir_in, dir_out)
squad2_csv(dir_in, dir_out)
copa_csv(dir_in, dir_out)

get_statistics(dir_out)