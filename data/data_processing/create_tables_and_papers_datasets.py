from collections import defaultdict

from argparse import ArgumentParser
import json

import pandas as pd
from tqdm import tqdm


def main():
    argp = ArgumentParser()
    argp.add_argument("in_tables_path", type=str)
    argp.add_argument("out_tables_path", type=str)
    argp.add_argument("out_papers_path", type=str)
    args = argp.parse_args()

    with open(args.in_tables_path) as f:
        in_tables = [json.loads(line.strip()) for line in f]

    bib_table_map = defaultdict(list)
    bib_entries = {}

    with open(args.out_tables_path, "w") as f:
        for table in tqdm(in_tables):
            table_json = {}
            table_json["tabid"] = table["_table_hash"]
            if not table["table_json"]["table_dict"]:
                print(f"Skipping {table['_table_hash']}")
                continue

            df = pd.DataFrame(table["table_json"]["table_dict"])
            # rewrite the "References" to have the *corpus_ids* rather than the *bib_hash* prefixes or *arxiv_ids*
            row_subset = []
            for row in table["row_bib_map"]:
                if row["corpus_id"] != -1:
                    df["References"][row["row"]] = row["corpus_id"]
                    row_subset.append(row["row"])
            df = df.iloc[row_subset].set_index("References")

            df = df.map(
                lambda x: x if isinstance(x, list) else [x]
            )  # this is a silly formatting thing - table values are lists. not sure if we'll keep it or not
            table_json["table"] = df.to_dict()
            table_json["row_bib_map"] = table["row_bib_map"]
            for row_entry in table["row_bib_map"]:
                bib_table_map[row_entry["corpus_id"]].append(table["_table_hash"])
                if row_entry["corpus_id"] != -1:
                    bib_entries[row_entry["corpus_id"]] = {
                        "title": row_entry["title"],
                        "abstract": row_entry["abstract"],
                    }
            table_json["caption"] = table["caption"]
            table_json["in_text_ref"] = table["in_text_ref"]
            table_json["arxiv_id"] = table["paper_id"]
            f.write(json.dumps(table_json) + "\n")

    # papers.jsonl

    print()
    with open(args.out_papers_path, "w") as f:
        num_skipped = 0
        for corpus_id in tqdm(bib_table_map):
            paper = {}
            if corpus_id not in bib_entries:
                continue
            try:
                paper["tabids"] = bib_table_map[corpus_id]
                paper["corpus_id"] = corpus_id
                paper["title"] = bib_entries[corpus_id]["title"]
                paper["abstract"] = bib_entries[corpus_id]["abstract"]

            except KeyError:
                print(bib_entries[corpus_id].keys())
            except AttributeError:
                num_skipped += 1
                continue
            f.write(json.dumps(paper) + "\n")
        print(f"Num skipped papers: {num_skipped}")


if __name__ == "__main__":
    main()
