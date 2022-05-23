import csv
import json
import jsonlines


def read_csv(path):
    lines = []
    with open(path, encoding='UTF8') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        l = [line for line in reader]
        lines += l
    return lines

def write_csv(path, lines, headers=None):
    if headers is None:
        headers = ["input", "output"] # default headers
    f = open(path, mode="w", encoding='UTF8', newline='')
    writer = csv.writer(f)
    writer.writerow(headers)
    for line in lines:
        writer.writerow(line)