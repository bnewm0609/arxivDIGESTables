from argparse import ArgumentParser
import csv
import glob
import gzip
import json
import os
from pathlib import Path
import shutil
import time
import re
from tqdm import tqdm


import time
import requests

DOWNLOAD_URL="" # contact the authors for the download url

def save_jsons(jsons, out_file):
    with open(out_file, "a") as f:
        for sample in jsons:
            f.write(json.dumps(sample) + "\n")


def main():
    argp = ArgumentParser()
    argp.add_argument("papers_file")
    argp.add_argument("--out_file")
    argp.add_argument("--start", type=int, default=0)
    argp.add_argument("--count", type=int, default=None)
    args = argp.parse_args()
    print("starting")

    data_jsons = []
    with open(args.papers_file) as f:
        papers = [json.loads(line) for line in f]

    start = args.start
    count = len(papers) - args.start if args.count is None else args.count
    papers = papers[start : start + count]

    # filter out corpus_ids that we've already downloaded
    try:
        with open(args.out_file) as f:
            obtained_corpus_ids = [json.loads(line)["metadata"]["corpusId"] for line in tqdm(f, total=2513)]
            # obtained_corpus_ids = {paper["metadata"]["corpusId"] for paper in previous_papers}
            assert obtained_corpus_ids
            print(list(obtained_corpus_ids[:10]), len(obtained_corpus_ids))

            papers = [paper for paper in papers if paper["corpus_id"] not in obtained_corpus_ids]
    except FileNotFoundError:
        pass

    unshowable_corpus_ids = []
    other_error_corpus_ids = []
    for sample in tqdm(papers):
        corpus_id = sample["corpus_id"]

        response = requests.get(f"{DOWNLOAD_URL}{corpus_id}")

        data = response.json()

        if "error" in data:
            # print(f"Skipping {corpus_id} due to error:")
            # print(data)
            if isinstance(data["error"], str) and data["error"].startswith("CorpusId is not showable"):
                unshowable_corpus_ids.append(corpus_id)
            else:
                other_error_corpus_ids.append(corpus_id)

            time.sleep(0.1)
            continue

        if "metadata" not in data:
            print(data.keys())
            data["metadata"] = {"corpusId": corpus_id}
        else:
            data["metadata"]["corpusId"] = corpus_id

        data_jsons.append(data)
        if len(data_jsons) >= 5:
            save_jsons(data_jsons, args.out_file)
            data_jsons = []

        time.sleep(0.5)

    save_jsons(data_jsons, args.out_file)
    save_jsons(
        [{"unshowable_ids": unshowable_corpus_ids, "other_error_ids": other_error_corpus_ids}],
        os.path.splitext(args.out_file)[0] + "_errors.jsonl",
    )
    print("done")


if __name__ == "__main__":
    main()
