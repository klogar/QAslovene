import deepl
import csv

auth_key = ""  # Replace with your key

def translate_deepl_mctest(texts, ids):
    # Create a Translator object providing your DeepL API authentication key.
    # Be careful not to expose your key, for example when sharing source code.

    translator = deepl.Translator(auth_key)
    # This example is for demonstration purposes only. In production code, the
    # authentication key should not be hard-coded, but instead fetched from a
    # configuration file or environment variable.

    result = translator.translate_text(texts, target_lang="SL")
    f_out = open(f"../../../Magistrska/Datasets/English/translationprep/MCTest-deepl.csv", 'a', encoding='UTF8', newline='')
    writerc = csv.writer(f_out)
    for id, mc in zip(ids, result):
        mc_split = mc.text.split(" | ")
        writerc.writerow([id] + mc_split)


    # Check account usage
    usage = translator.get_usage()
    if usage.character.limit_exceeded:
        print("Character limit exceeded.")
    else:
        print(f"Character usage: {usage.character.count} of {usage.character.limit}")

def translate_deepl_squad(texts, ids, type):
    # Create a Translator object providing your DeepL API authentication key.
    # Be careful not to expose your key, for example when sharing source code.
    translator = deepl.Translator(auth_key)
    # This example is for demonstration purposes only. In production code, the
    # authentication key should not be hard-coded, but instead fetched from a
    # configuration file or environment variable.

    result = translator.translate_text(texts, target_lang="SL")
    f_out = open(f"../../../Magistrska/Datasets/English/translationprep/SQUAD-deepl-{type}.csv", 'a', encoding='UTF8', newline='')
    writerc = csv.writer(f_out)
    for id, mc in zip(ids, result):
        mc_split = mc.text.split(" | ")
        writerc.writerow([id] + mc_split)


    # Check account usage
    usage = translator.get_usage()
    if usage.character.limit_exceeded:
        print("Character limit exceeded.")
    else:
        print(f"Character usage: {usage.character.count} of {usage.character.limit}")

def read_dataset(filename):
    dataset = []
    with open(f"{filename}", encoding='UTF8') as f:
        csv_file = csv.reader(f)
        dataset = [line for line in csv_file]

    def concat_text(line):
        return " | ".join(line[1:])

    ids = list(map(lambda x: x[0], dataset))[:700]
    dataset = list(map(concat_text, dataset[:700]))
    return dataset, ids

# mctest = read_dataset("../../../Magistrska/Datasets/English/translationprep/MCTest.csv")
# translate_deepl_mctest(*mctest)
# squad = read_dataset("../../../Magistrska/Datasets/English/translationprep/Squad-train-qa.csv")
# translate_deepl_squad(*squad, "train-qa")
# squad = read_dataset("../../../Magistrska/Datasets/English/translationprep/Squad-val-qa.csv")
# translate_deepl_squad(*squad, "val-qa")

def compare_translations_mctest(deepl_file, my_file):
    with open(deepl_file, encoding='UTF8') as f1, open(my_file, encoding="UTF8") as f2:
        csv_file = csv.reader(f1)
        deep = [line for line in csv_file]
        csv_file = csv.reader(f2)
        my = [line for line in csv_file]

    for d,m in zip(deep, my):
        for i in range(4):
            print(f"DEEPL: {d[2+i*5:2+i*5+5]}")
            print(f"MY: {m[2+i*5:2+i*5+5]}")

def compare_translations_squad(deepl_file, my_file):
    with open(deepl_file, encoding='UTF8') as f1, open(my_file, encoding="UTF8") as f2:
        csv_file = csv.reader(f1)
        deep = [line for line in csv_file]
        csv_file = csv.reader(f2)
        my = [line for line in csv_file]

    for d,m in zip(deep, my):
        print(f"DEEPL: {d}")
        print(f"MY: {m}")

# compare_translations_mctest("../../../Magistrska/Datasets/English/translationprep/MCTest-deepl.csv", "../../../Magistrska/Datasets/MT/translationprep/MCTest-sl.csv")
# compare_translations_squad("../../../Magistrska/Datasets/English/translationprep/Squad-deepl-train-qa.csv", "../../../Magistrska/Datasets/MT/translationprep/Squad-train-qa-sl.csv")

