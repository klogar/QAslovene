import os
import csv
import random

fieldnames = ["input", "output"]
random.seed(30)

def merge(kind):

    dir = './encoded'
    directories = [os.path.join(dir, o) for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o))]
    lines = []
    for directory in directories:
        with open(f"{directory}/{kind}.csv", encoding='UTF8') as f:
            reader = csv.reader(f)
            next(reader, None) # skip header
            l = [line for line in reader]
            lines += l
    random.shuffle(lines)
    f = open(f"{kind}.csv", mode="w", encoding='UTF8', newline='')
    writer = csv.writer(f)
    writer.writerow(["input", "output"])
    for line in lines:
        writer.writerow(line)


kinds = ["train", "val", "test_answered"]
for kind in kinds:
    merge(kind)