import csv
from pathlib import Path

INPUT_PATH = "safety_probe_prompts.csv"
OUTPUT_FOLDER = Path("data")


print(OUTPUT_FOLDER)

with open(INPUT_PATH, encoding="utf-8") as input_file:
    reader = csv.DictReader(input_file)
    rows = list(reader)

# create 3 empty list

train_data=[]
test_iid_data=[]
test_shift_data=[]

for row in rows:
    prompt = row["text"]
    label = row["label"]
    split = row["split"]

    sample = [prompt, label]

    if split == "train":
        train_data.append(sample)
    elif split == "test_iid":
        test_iid_data.append(sample)
    elif split == "test_shift":
        test_shift_data.append(sample)

def write_file(name_file, data):
    with open(
        name_file,
        mode="w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file, delimiter="|")
        writer.writerows(data)

write_file("train.txt", train_data)
write_file("test_shift.txt", test_shift_data)
write_file("test_iid.text",test_iid_data)