from argparse import ArgumentParser
import json
from tqdm import tqdm

from metrics import SchemaRecallMetric
from metrics_utils import (
    BaseFeaturizer,
    DecontextFeaturizer,
    ExactMatchScorer,
    JaccardAlignmentScorer,
    Llama3AlignmentScorer,
    SentenceTransformerAlignmentScorer,
    ValueFeaturizer,
)
from table import Table

def open_gold_tables(tables_path):
    """
    Returns a mapping from tabid to gold Table objects
    """

    tabid_to_gold_table = {}
    with open(tables_path) as f:
        for line in f:
            table_dict = json.loads(line)
            tabid = table_dict["tabid"]
            table = Table(
                tabid=tabid,
                schema=list(table_dict["table"].keys()),
                values=table_dict["table"],
                caption=table_dict["caption"],
            )
            tabid_to_gold_table[tabid] = table
    return tabid_to_gold_table

def open_pred_tables(tables_path):
    pred_tables = []
    with open(tables_path) as f:
        for line in f:
            table_dict = json.loads(line)
            pred_table = Table(
                tabid=table_dict["metadata"]["tabid"],
                schema=list(table_dict["table"].keys()),
                values=table_dict["table"]
            )
            table_dict["table_cls"] = pred_table
            pred_tables.append(table_dict)
    return pred_tables

def load_featurizer(featurizer_name):
    if featurizer_name == "name":
        return BaseFeaturizer("name")
    elif featurizer_name == "values":
        return ValueFeaturizer("values")
    elif featurizer_name == "decontext":
        return DecontextFeaturizer("decontext", model="mistralai/Mixtral-8x7B-Instruct-v0.1")
    else:
        raise ValueError(f"Unknown featurizer name: {featurizer_name}.")

def load_scorer(scorer_name):
    if scorer_name == "exact_match":
        return ExactMatchScorer()
    elif scorer_name == "jaccard":
        return JaccardAlignmentScorer(remove_stopwords=True)
    elif scorer_name == "sentence_transformers":
        return SentenceTransformerAlignmentScorer()
    elif scorer_name == "llama3":
        return Llama3AlignmentScorer()

def main():
    argp = ArgumentParser()
    argp.add_argument("--gold_tables", type=str)
    argp.add_argument("--pred_tables", type=str)
    argp.add_argument("--out_file", type=str)
    argp.add_argument("--featurizer", type=str, default="decontext", choices=["name", "values", "decontext"], help="name: uses the column name; values: concatenates the column name with the column's values; decontext: decontextualizes the column name using the values as context.")
    argp.add_argument("--scorer", type=str, default="sentence_transformers", choices=["exact_match", "jaccard", "sentence_transformers", "llama3"])
    argp.add_argument("--threshold", type=float, default=0.7, help="Threshold used to determine a match for exact_match, jaccard and sentence_transformer scorers")
    argp.add_argument("--eval_type", type=str, default="schema", choices=["schema", "values"])
    args = argp.parse_args()

    # open gold and predicted tables
    tabid_to_gold_table = open_gold_tables(args.gold_tables)
    pred_tables = open_pred_tables(args.pred_tables)

    # load the metric
    featurizer = load_featurizer(args.featurizer)
    scorer = load_scorer(args.scorer)
    metric = SchemaRecallMetric(featurizer=featurizer, alignment_scorer=scorer, sim_threshold=args.threshold)

    # run the evaluation
    results = []
    for pred_table_instance in tqdm(pred_tables):
        pred_table = pred_table_instance.pop("table_cls")
        gold_table = tabid_to_gold_table[pred_table.tabid]
        recall, _, alignment = metric.add(pred_table, gold_table, return_scores=True)
        alignment_str_keys = dict(zip(map(str, alignment), alignment.values()))
        results.append(pred_table_instance | {"scores": {"recall": recall, "alignment": alignment_str_keys, "featurizer": args.featurizer, "scorer": args.scorer, "threshold": args.threshold}})
    
    # write the results to disk
    with open(args.out_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
        
    scores_dict = metric.process_scores()




if __name__ == "__main__":
    main()