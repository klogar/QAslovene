import jsonlines
import pandas as pd

models = [
    ("unified-general-all-3", 345030),
    ("unified-without-noanswer-all", 259279),
    ("unified-without-noanswer-based", 345030),
    ("unified-lower-all", 361460),
    ("munified", 361460),
    ("munified-equal", 534896),
    ("munified-slo-engall", 1061880),
    ("munified-eng-slo", 312170),
    ("munified-engall-slo", 345030)
]

kind = "test_answered"
# datasets = ["COPA", "MCTest", "MultiRC", "SQUAD2-project"]
# datasets = ["MCTest-eng"]
# datasets = ["SQUAD2-project"]
datasets = ["COPA"]
datasets_mc = ["COPA", "MCTest", "MCTest-eng"]

lang = "slo" # eng or slo

def get_predictions(dataset):
    predictions_dict = dict()
    if lang == "slo":
        for model, checkpoint in models:

            prediction_file = f"../models/{model}/checkpoint-{checkpoint}/{dataset}_{kind}_generated_predictions.txt"

            with open(prediction_file) as f:
                predictions = [line.strip() for line in f.readlines()]

            predictions_dict[model] = predictions

    elif lang == "eng":
        prediction_file = f"../predictions/English-lower/{dataset}_{kind}_generated_predictions.txt"
        with open(prediction_file) as f:
            predictions = [line.strip() for line in f.readlines()]

        predictions_dict["unifiedqa"] = predictions

    return predictions_dict

def get_answers(dataset):
    reader = jsonlines.open(f"./../datasets/encoded/answers/{dataset}-{kind}.jsonl")
    golds = [line["answers"] for line in reader]
    return golds

def get_questions_context_answers(dataset):
    data = pd.read_csv(f"../datasets/encoded/{dataset}/{kind}.csv")
    input_lines = list(data["input"])
    answers = list(data["output"])
    questions = [input.split("\\n")[0] for input in input_lines]
    context = [input.split("\\n")[1] for input in input_lines]
    return questions, context, answers

def get_answers_questions_options_context_mc(dataset):
    data = pd.read_csv(f"../datasets/encoded/{dataset}/{kind}.csv")
    golds = list(data["output"])
    input_lines = list(data["input"])
    questions = [input.split("\\n")[0] for input in input_lines]
    options = [input.split("\\n")[1] for input in input_lines]
    context = [input.split("\\n")[2] for input in input_lines]
    return golds, questions, options, context


def get_qual_eval():
    predictions = dict()
    for dataset in datasets:
        predictions[dataset] = get_predictions(dataset)

    answers = dict()
    questions = dict()
    options = dict()
    contexts = dict()
    for dataset in datasets:
        if dataset in datasets_mc:
            ans, qs, opts, cnts = get_answers_questions_options_context_mc(dataset)
            answers[dataset] = ans
            questions[dataset] = qs
            options[dataset] = opts
            contexts[dataset] = cnts
        else:
            qs, cnts, ans = get_questions_context_answers(dataset)
            if dataset in ["MultiRC", "SQUAD2-project"]: # there are multiple correct answers, read from a specialized file
                ans = get_answers(dataset)
            answers[dataset] = ans
            questions[dataset] = qs
            contexts[dataset] = cnts

    for dataset in datasets:
        preds = predictions[dataset]
        for ind, (answer, question) in enumerate(zip(answers[dataset], questions[dataset])):
            print(f"QUESTION: {question}")
            print(f"ANSWER: {answer}")
            if dataset in datasets_mc:
                print(f"OPTIONS: {options[dataset][ind]}")
            print(f"Model results:")
            if lang == "slo":
                for model, _ in models:
                    p = preds[model][ind]
                    # print(f"{model}: {p} ------- {p.lower() in options[dataset][ind].lower() or '< Ni odgovora >' in answer}")
                    # print(f"{model}: {p} ------- {p.lower() in contexts[dataset][ind].lower() or '< Ni odgovora >' in answer}")
                    print(f"{model}: {p} ------- {p == answer}")
            elif lang == "eng":
                p = preds["unifiedqa"][ind]
                # print(f"unifiedqa: {p} ------- {p.lower() in options[dataset][ind].lower() or '< Ni odgovora >' in answer}")
                print(f"unifiedqa: {p} ------- {p.lower() in contexts[dataset][ind].lower() or '< Ni odgovora >' in answer}")
            print()



if __name__ == "__main__":
    get_qual_eval()