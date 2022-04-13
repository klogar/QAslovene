import json
import jsonlines

def get_statistics(dir, datasets, kinds):
    for dataset in datasets:
        for kind in kinds:
            with open(f"{dir}/HT/{dataset}/{kind}.jsonl", encoding="utf-8") as f:
                ht_lines = len(f.readlines())
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

def merge(datasets, kinds):
    for dataset in datasets:
        for kind in kinds:
            data_ht = []
            dataset_name = dataset if dataset != "test" else "test_answered"
            with jsonlines.open("HT/" + dataset_name + "/" + kind + ".jsonl") as reader:
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

def load_multirc_fixedids(dir):

    with open(f"{dir}/MT/testprep/test2.json") as f:
        data = json.load(f)["data"]
        print()


# dir = "C:/Users/Katja/Documents/FRI/Magistrska/Datasets"
datasets = ["BoolQ", "COPA", "MultiRC"]
kinds = ["train", "val", "test"]
dir = "../../../Magistrska/Datasets"
get_statistics(dir, datasets, kinds)
# merge(datasets, kinds)

# load_multirc_fixedids(dir)