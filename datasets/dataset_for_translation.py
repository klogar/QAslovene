import json
import jsonlines
import csv
import re

def mctest(dir_in, dir_out):
    kinds = ["train", "dev", "test"]
    sizes = [160,500]
    f_out = open(f"{dir_out}/MCTest.csv", 'w', encoding='UTF8', newline='')
    # fieldnames = ["id", "context", "question1", "answer11", "answer12", "answer13", "answer14"
    #                                 "question2", "answer21", "answer22", "answer23", "answer24"
    #                                 "question3", "answer31", "answer32", "answer33", "answer34"
    #                                 "question4", "answer41", "answer42", "answer43", "answer44"]
    writer = csv.writer(f_out)
    for kind in kinds:
        for size in sizes:
            with open(f"{dir_in}/MCTest/mc{size}.{kind}.tsv") as f:
                tsv_file = csv.reader(f, delimiter="\t")
                for ind, line in enumerate(tsv_file):
                    line_out = [f"{size}{kind[:2]}{ind}"]
                    for part in line[2:]:
                        part = part.replace("\t", "").replace("\\newline", " ").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                        if part.startswith("one:"):
                            part = part.replace("one:", "")
                        elif part.startswith("multiple:"):
                            part = part.replace("multiple:", "")
                        line_out.append(part)
                    writer.writerow(line_out)

def squad(dir_in, dir_out):
    kinds = ["train", "val"]
    for kind in kinds:
        f_out = open(f"{dir_out}/Squad-{kind}-qa.csv", 'w', encoding='UTF8', newline='')
        writer = csv.writer(f_out)
        f_outc = open(f"{dir_out}/Squad-{kind}-context.csv", 'w', encoding='UTF8', newline='')
        writerc = csv.writer(f_outc)
        counter = 0
        with open(f"{dir_in}/SQUAD2/{kind}.json") as f:
            data = json.load(f)["data"]
            for d in data:
                for paragraph in d["paragraphs"]:
                    counter += 1
                    writerc.writerow([counter, d["title"].replace("_", " ") + " " + paragraph["context"]])
                    for qas in paragraph["qas"]:
                        row = [counter, qas["question"]]
                        if qas["is_impossible"]:
                            row.append("<No Answer>")
                        else:
                            #ali je lahko več kot en odgovor
                            row.append(qas["answers"][0]["text"])
                        writer.writerow(row)

def mctest_qa(dir_in, dir_out):
    f_out = open(f"{dir_out}/MCTest-sl-qa.csv", 'w', encoding='UTF8', newline='')
    writer = csv.writer(f_out)
    with open(f"{dir_in}/MCTest-sl.csv", encoding='UTF8') as f:
        csv_file = csv.reader(f)
        for line in csv_file:
            writer.writerow(list(map(lambda x: x[:-1] if x.endswith(".") else x, line[2:]))) # Remove dot

def mctest_read_answers(dir_in, kind):
    sizes = [160, 500]
    all_answers = []
    for size in sizes:
        path = f"{dir_in}/mc{size}.{kind}.ans"
        with open(path.replace(".tsv", ".ans")) as f:
            for l in f.readlines():
                all_answers.extend(l.replace("\n", "").split("\t"))
    return all_answers

def get_answer(line, ans):
    if ans == "A":
        return line[2]
    elif ans == "B":
        return line[3]
    elif ans == "C":
        return line[4]
    elif ans == "D":
        return line[5]

def mctest_split(dir_in, dir_out):
    train = []
    val = []
    test = []
    with open(f"{dir_in}/MCTest-sl-8.csv", encoding='UTF8') as f:
        csv_file = csv.reader(f)
        for line in csv_file:

            if "tr" in line[0]:
                dataset = train
            elif "de" in line[0]:
                dataset = val
            elif "te" in line[0]:
                dataset = test

            for ind in range(2, 22, 5):
                dataset.append([line[1],line[ind],line[ind+1],line[ind+2],line[ind+3],line[ind+4]])

    train_ans = mctest_read_answers(dir_in, "train")
    val_ans = mctest_read_answers(dir_in, "val")
    test_ans = mctest_read_answers(dir_in, "test")

    f_out = open(f"{dir_out}/MCTest/train.csv", 'w', encoding='UTF8', newline='')
    writer = csv.writer(f_out)
    for line, ans in zip(train, train_ans):
        line.append(get_answer(line,ans))
        writer.writerow(line)

    f_out = open(f"{dir_out}/MCTest/val.csv", 'w', encoding='UTF8', newline='')
    writer = csv.writer(f_out)
    for line, ans in zip(val, val_ans):
        line.append(get_answer(line,ans))
        writer.writerow(line)

    f_out = open(f"{dir_out}/MCTest/test.csv", 'w', encoding='UTF8', newline='')
    writer = csv.writer(f_out)
    for line, ans in zip(test, test_ans):
        line.append(get_answer(line,ans))
        writer.writerow(line)

def squad2_context_to_qa(contexts, qa):
    lines = []
    for id, question, answer in qa:
        for idc, context in contexts:
            if id == idc:
                lines.append([context, question, answer])
                break
    return lines

def squad2_split(dir_in, dir_out):
    for kind in ["train", "val"]:
        f_c = open(f"{dir_in}/Squad-{kind}-context-sl.csv", encoding='UTF8')
        contexts = csv.reader(f_c)
        contexts = [line for line in contexts]
        f_qa = open(f"{dir_in}/Squad-{kind}-qa-sl.csv", encoding='UTF8')
        qa = csv.reader(f_qa)
        qa = [line for line in qa]
        if kind == "train":

            train_test_split = 3500
            contexts_train = contexts[:-train_test_split]
            contexts_test = contexts[-train_test_split:]

            f_out = open(f"{dir_out}/SQUAD2/train.csv", 'w', encoding='UTF8', newline='')
            writer = csv.writer(f_out)
            for line in squad2_context_to_qa(contexts_train, qa):
                writer.writerow(line)

            f_out = open(f"{dir_out}/SQUAD2/test.csv", 'w', encoding='UTF8', newline='')
            writer = csv.writer(f_out)
            for line in squad2_context_to_qa(contexts_test, qa):
                writer.writerow(line)

        else:
            f_out = open(f"{dir_out}/SQUAD2/val.csv", 'w', encoding='UTF8', newline='')
            writer = csv.writer(f_out)
            for line in squad2_context_to_qa(contexts, qa):
                writer.writerow(line)





dir_in = "../../../Magistrska/Datasets/English/original"
dir_out = "../../../Magistrska/Datasets/English/translationprep"

# mctest(dir_in, dir_out)
# squad(dir_in, dir_out)

dir_in_split = "../../../Magistrska/Datasets/MT/translationprep"
dir_out_split = "../../../Magistrska/Datasets/ALL"

# mctest_qa(dir_in_split, dir_in_split)
mctest_split(dir_in_split, dir_out_split)
squad2_split(dir_in_split, dir_out_split)

