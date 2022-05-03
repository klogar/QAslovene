import pandas as pd
import datasets
import multirc
import jsonlines

# dataset_names = ["BoolQ", "COPA", "MCTest", "MultiRC", "SQUAD2"]
dataset_names = ["MultiRC"]
rouge = datasets.load_metric("rouge")
squad = datasets.load_metric("squad")
mrc = datasets.load_metric("multirc.py")

for dataset in dataset_names:
    test_file = f"./datasets/encoded/{dataset}/test_answered.csv"
    # test_file = f"./datasets/test_answered.csv"
    test_data = pd.read_csv(test_file)
    answers = list(test_data["output"])

    # if dataset == "MultiRC":
    #     dataset = "MultiRC-small"

    # prediction_file = f"./models/{dataset}/generated_predictions.txt" # for each model by itself
    prediction_file = f"./predictions/{dataset}/generated_predictions.txt" # for common model unified
    with open(prediction_file) as f:
        predictions = [line.strip() for line in f.readlines()]

    print(f"{dataset}")


    rouge.add_batch(predictions=predictions, references=answers)
    results = rouge.compute()
    print(f"rouge: {results['rougeL'].mid.fmeasure:.2f}")

    p = list(map(lambda x: {"prediction_text": x[1], "id": str(x[0])}, enumerate(predictions)))
    a = list(map(lambda x: {"answers": {'answer_start': [0], "text": [x[1]]}, "id": str(x[0])}, enumerate(answers)))
    squad.add_batch(predictions=p, references=a)
    results = squad.compute()
    print(f"f1: {results['f1']:.2f}")
    print(f"exact: {results['exact_match']:.2f}")

    if dataset == "BoolQ":
        print(f"All non da/ne answers: {list(filter(lambda x: x not in ['da', 'ne'], predictions))}")
    if dataset == "MultiRC":
        with jsonlines.open(f"./datasets/encoded/answers/MultiRC.jsonl") as reader:
            answers = [line["answers"] for line in reader]
        mrc.add_batch(predictions=predictions, references=answers)
        results = mrc.compute()
        print(f"rouge-all-answers: {results:.2f}")
    if dataset == "SQUAD2":
        ap = list(zip(answers, predictions))

        print(f"All incorrect <no answer>: {len(list(filter(lambda x: x[0] != '< Ni odgovora >' and x[1] == '< Ni odgovora >', ap)))/len(predictions)}")
        print(f"All missing <no answer>: {len(list(filter(lambda x: x[0] == '< Ni odgovora >' and x[1] != '< Ni odgovora >', ap)))/len(predictions)}")
        print(f"Actual <no answer>: {len(list(filter(lambda x: x == '< Ni odgovora >', answers))) / len(answers)}")
        print(f"All predicted <no answer>: {len(list(filter(lambda x: x == '< Ni odgovora >', predictions))) / len(predictions)}")

        ap = list(filter(lambda x: x[0] != '< Ni odgovora >', ap))

        p = list(map(lambda x: {"prediction_text": x[1], "id": str(x[0])}, enumerate(ap[1])))
        a = list(map(lambda x: {"answers": {'answer_start': [0], "text": [x[1]]}, "id": str(x[0])}, enumerate(ap[0])))
        rouge.add_batch(predictions=p, references=a)
        results = rouge.compute()
        print(f"rouge: {results['rougeL'].mid.fmeasure:.2f}")
        squad.add_batch(predictions=p, references=a)
        results = squad.compute()
        print(f"f1: {results['f1']:.2f}")
        print(f"exact: {results['exact_match']:.2f}")
    else:
        print(f"All incorrect <no answer>: {len(list(filter(lambda x: x == '< Ni odgovora >', predictions))) / len(predictions)}")


    print()

