"""
Example command to run this script:
python scripts/data_processing/data_editor.py data/v3/highest_quality_tables_1k/dataset.jsonl bcb47c61-2523-4daa-a059-9288c56b21e1 --out_file data/xml_to_json_gold_data/dataset.jsonl
"""

from argparse import ArgumentParser
import json
import sys
import shutil

import dtale
import pandas as pd


def main():
    argp = ArgumentParser()
    argp.add_argument("input_file", type=str)
    argp.add_argument("table_hash", type=str)
    argp.add_argument("--out_file", type=str, default="data/xml_to_json_gold_data/dataset.jsonl")
    args = argp.parse_args()

    # data_path = "data/v3/highest_quality_tables_1k/dataset.jsonl"
    table = None
    with open(args.input_file) as f:
        for line in f:
            json_line = json.loads(line)
            if json_line["_table_hash"] == args.table_hash:
                table = json_line
                break

    if table is None:
        print(
            f"Unable to find table with table hash {args.table_hash} in the provided input_file: {args.input_file}"
        )
        sys.exit(0)

    print(table["paper_id"])
    print(f"https://www.arxiv.org/pdf/{table['paper_id']}")
    print(table["caption"])
    # to show:
    dt = dtale.show(pd.DataFrame(table["table_json"]["table_dict"]))
    dt.open_browser()

    choice = ""
    while choice != "Y" and choice != "n":
        choice = input("Ready to save? (Y/n) > ").strip()

    if choice == "Y":
        data_dict = dt.data.to_dict(orient="list")
        data_dict = {key.strip(): [val.strip() for val in vals] for key, vals in data_dict.items()}
        print(data_dict)

        with open(args.out_file) as f:
            gold_dataset = [json.loads(line) for line in f]

        for i in range(len(gold_dataset)):
            if gold_dataset[i]["_table_hash"] == args.table_hash:
                gold_dataset[i]["table_json"]["table_dict"] = data_dict
                break
        else:
            table["table_json"]["table_dict"] = data_dict
            gold_dataset.append(table)

        shutil.copy(args.out_file, args.out_file + ".bak")
        with open(args.out_file, "w") as f:
            for line in gold_dataset:
                f.write(json.dumps(line) + "\n")
        print(f"Saved! (Now {len(gold_dataset)} total gold examples)")
    else:
        print("Exited without saving")


if __name__ == "__main__":
    main()
