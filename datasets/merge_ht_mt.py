import json
import jsonlines

def get_statistics(datasets, kinds):
    for dataset in datasets:
        for kind in kinds:
            with open("HT/" + dataset + "/" + kind + ".jsonl", encoding="utf-8") as f:
                ht_lines = len(f.readlines())
            with open("MT/" + dataset + "/" + kind + ".jsonl", encoding="utf-8") as f:
                mt_lines = len(f.readlines())
            with open("ALL/" + dataset + "/" + kind + ".jsonl", encoding="utf-8") as f:
                all_lines = len(f.readlines())
            print(f"{dataset} {kind} human translated {ht_lines}, machine translated {mt_lines}, all {all_lines}")

def find_property_with_value(array, property, value):
    for item in array:
        if item[property] == value:
            return True
    return False

def merge(datasets, kinds):
    for dataset in datasets:
        for kind in kinds:
            data_ht = []
            with jsonlines.open("HT/" + dataset + "/" + kind + ".jsonl") as reader:
                for line in reader:
                    data_ht.append(line)
            data_mt = []
            # Only add those machine translated lines that are not in human translated lines
            with jsonlines.open("MT/" + dataset + "/" + kind + ".jsonl") as reader:
                for line in reader:
                    if not find_property_with_value(data_ht, "idx", line["idx"]):
                        data_mt.append(line)
            data = data_ht + data_mt
            with jsonlines.open("ALL/" + dataset + "/" + kind + ".jsonl", mode="w") as writer:
                for line in data:
                    writer.write(line)


# dir = "C:/Users/Katja/Documents/FRI/Magistrska/Datasets"
datasets = ["BoolQ", "COPA", "MultiRC"]
kinds = ["train", "val", "test"]
get_statistics(datasets, kinds)
# merge(datasets, kinds)