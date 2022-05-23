from my_utils import read_csv
from utils import make_input_string
import random

dir_in = "./../encoded"
dir_out = "."
kinds = ["train", "val", "test_answered"]

all_possible_angles =

def encode_qa(dataset_name, randomly_sample_angles=False):
    for kind in kinds:
        filename = f"{dir_in}/{dataset_name}/{kind}.csv"
        lines = read_csv(filename)

        data = []
        for line in lines:
            question, context = line["input"].strip().split("\n")
            answer = line["output"]
            data.append({
                "Q": question,
                "A": answer,
                "C": context
            })

        possible_angles = ["QC->A", "AC->Q"]

        for datum in data:
            get_data_in_slots(datum, possible_angles, randomly_sample_angles)

def get_data_in_slots(data, possible_angles, randomly_sample_angles):
    if randomly_sample_angles: # if we should only consider one angle for training (only for huge datasets such as SQUAD and MultiRC)
        angles = random.choice(possible_angles)
    else:
        angles = possible_angles
    for angle in angles:
        input_slots, output_slots = angle.split("->")
        make_input_string(data)




def encode_mcqa():
    pass



encode_qa("BoolQ")