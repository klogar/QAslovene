import pandas as pd
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from eval_utils import *

kinds = ["train", "val", "test_answered"]
options = ["A", "B", "C", "D"]

def plot_mc(data, num_options, dataset):
    df = pd.DataFrame(data, columns=options[:num_options], index=["učna", "validacijska", "testna"])
    df.plot(kind='bar',alpha=0.75, rot=0)
    plt.ylabel("Odstotek vseh primerov")
    plt.title(f"Razredna distribucija v podatkovni množici {dataset}")
    # plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.show()

def mc_info(dataset):
    splits = defaultdict(list)
    for kind in kinds:
        file = f"{dir}/{dataset}/{kind}.csv"
        data = pd.read_csv(file)
        gold_lines = list(data["output"])
        input_lines = list(data["input"])

        regex = re.compile("\([A-E]\)")

        split = defaultdict(int)
        all = len(gold_lines)
        for input, gold in zip(input_lines, gold_lines):
            input_split = input.split("\\n")
            candidates_string = input_split[1].strip()
            candidates_split = regex.split(candidates_string)
            candidates_split = [x.strip() for x in candidates_split if len(x.strip()) > 0]
            split[candidates_split.index(gold)] += 1

        # splits[kind] = list(map(lambda x: 100*x[1]/all, sorted(split.items())))
        print(f"{dataset} - {kind}")
        for key, value in sorted(split.items()):
            splits[options[key]].append(100*value/all)
            print(f"-------> {key} = {100*value/all:.2f}")

    num_options = len(candidates_split)
    plot_mc(splits, num_options, dataset)

def copa_info():
    for kind in kinds:
        file = f"{dir}/COPA/{kind}.csv"
        data = pd.read_csv(file)
        gold_lines = list(data["output"])
        input_lines = list(data["input"])

        questions = [line.split("\\n")[0] for line in input_lines]
        counter = Counter(questions)
        print(counter)

def multirc_info():
    for kind in kinds:
        file = f"{dir}/MultiRC/{kind}.csv"
        data = pd.read_csv(file)
        gold_lines = list(data["output"])
        input_lines = list(data["input"])

        yes_no_lines = list(filter(lambda x: normalize_answer(x) in ["da", "ne"], gold_lines))
        counter = Counter(yes_no_lines)
        print(counter)

        print(f"{kind} - {100*len(yes_no_lines)/len(gold_lines):.1f} % of yes/no questions")

def squad2_info():
    for kind in kinds:
        file = f"{dir}/SQUAD2-project/{kind}.csv"
        data = pd.read_csv(file)

        gold_lines = list(data["output"])
        no_answers = list(filter(lambda x: x == "< Ni odgovora >", gold_lines))
        print(f"{kind} - % of no answers {len(no_answers)/len(gold_lines)}")

        type_lines = list(data["type"])
        ht = list(filter(lambda x: x == "HT", type_lines))
        print(f"{kind} - % of human translated {len(ht) / len(gold_lines)}, total {len(ht)}")

dir = "encoded"
# mc_info("COPA")
# mc_info("MCTest")
# copa_info()
# multirc_info()
squad2_info()