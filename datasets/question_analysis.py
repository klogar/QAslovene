import classla
import pandas as pd

def get_question_word(question):
    for word in question.to_dict()[0][0]:
        if word.get("xpos").startswith("Pq"):
            return word
    else:
         return f"ERROR: {question.to_dict()[0][0][0]}"

def get_root(text):
    for word in text.to_dict()[0][0]:
        if word.get("deprel") == "root":
            return word
    else:
         return f"ERROR"

def question(dataset):
    for kind in kinds:
        data = pd.read_csv(f"../datasets/encoded/{dataset}/{kind}.csv")
        input_lines = list(data["input"])
        answers = list(data["output"])[:50]
        questions = [input_line.split("\\n")[0] for input_line in input_lines][:50]
        for question, answer in zip(questions, answers):
            q = nlp(question)
            a = nlp(answer)
            # print(get_question_word(q))
            print(f"{get_root(a).get('text')} | {answer} | {question}")
            # print(q)
            # print(a)

dir = "encoded"
classla.download("sl")
# nlp = classla.Pipeline("sl", processors="tokenize,pos,lemma")
nlp = classla.Pipeline("sl")
kinds = ["train", "val", "test_answered"]

question("MCTest")