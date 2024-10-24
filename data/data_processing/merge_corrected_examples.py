"""Short script for merging in manually edited tables"""
from argparse import ArgumentParser
import json


def main():
    argp = ArgumentParser()
    argp.add_argument("--in_file", type=str)
    argp.add_argument("--gold_file", type=str, default="data/xml_to_json_gold_data/dataset.jsonl")
    argp.add_argument("--out_file", type=str)
    args = argp.parse_args()

    with open(args.in_file) as f:
        in_file = [json.loads(line) for line in f]

    with open(args.gold_file) as f:
        gold_data = [json.loads(line) for line in f]
        gold_data = {gold_tab["_table_hash"]: gold_tab for gold_tab in gold_data}

    with open(args.out_file, "w") as f:
        for sample in in_file:
            corrected_sample = gold_data.get(sample["_table_hash"], sample)
            f.write(json.dumps(corrected_sample) + "\n")


if __name__ == "__main__":
    main()
