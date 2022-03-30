import json
import jsonlines

def boolq(dir_in, dir_out, dataset, kinds):
    for kind in kinds:
        writer = jsonlines.open(f"{dir_out}/{dataset}/{kind}.jsonl", mode="w")
        with jsonlines.open(f"{dir_in}/{dataset}/{kind}.jsonl") as reader:
            for line in reader:
                line_out = dict()
                question = line["question"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                if '?' not in question:
                    question = question + "?"
                paragraph = line["passage"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                line_out["input"] = question + " \n " + paragraph
                if kind != "test": # test is without answers
                    answer = "yes" if line["label"] else "no"
                    line_out["output"] = answer
                writer.write(line_out)

def multirc(dir_in, dir_out, dataset, kinds):
    for kind in kinds:
        writer = jsonlines.open(f"{dir_out}/{dataset}/{kind}.jsonl", mode="w")
        with jsonlines.open(f"{dir_in}/{dataset}/{kind}.jsonl") as reader:
            for line in reader:
                paragraph = line["passage"]["text"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                for q in line["passage"]["questions"]:
                    question = q["question"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                    line_out = dict()
                    line_out["input"] = question + " \n " + paragraph
                    if kind != "test":
                        for a in q["answers"]:
                            if a["label"] == 1:
                                line_out["output"] = a["text"].replace("\t", "").replace("   ", " ").replace("  ", " ").replace("\n", " ")
                                break
                    writer.write(line_out)


kinds = ["train", "val", "test"]
dir_in = "ALL"
dir_out = "encoded"
boolq(dir_in, dir_out, "BoolQ", kinds)
multirc(dir_in, dir_out, "MultiRC", kinds)