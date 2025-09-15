import json
from pathlib import Path

from hydra.utils import to_absolute_path
from omegaconf import DictConfig


def convert_paths_to_absolute(config: DictConfig):
    """
    Turn any path in the passed config to an absolute path by recursing down the tree defined by
    the config yaml files by modifying the passed config in-place.
    """
    to_resolve = []
    for key in config:
        if isinstance(config[key], DictConfig):
            to_resolve.append(config[key])  # check if there's anything to resolve in the sub-config
        elif "path" in key:
            config[key] = to_absolute_path(config[key])
    for sub_config in to_resolve:
        convert_paths_to_absolute(sub_config)


def get_results_path(args: DictConfig) -> str:
    results_path = ""
    if args.mode == "baseline":
        # results_path = f"results/{args.data._id}/{args.endtoend.difficulty}/{args.endtoend.model_type}/{args.mode}/{args.endtoend.retry_type}"
        results_path = f"results/{args.data._id}"
    elif args.mode == "ours":
        results_path = (
            # f"results/{args.mode}/{args.data._id}__{args.attribute_gen_method}_popu{args.paper_loop}"
            f"results/{args.data._id}"
        )
    return results_path

def load_outputs(args: DictConfig):
    # load the tables: if the tables are stored in a jsonl file, then we can just read them in
    results_path = Path(args.results_path)
    print(results_path)
    if not results_path.exists():
        print(f"Results path {results_path} does not exist.")
        tables = []
    else:
        if not (results_path / "tables.jsonl").exists():
            print(f"Tables file {results_path / 'tables.jsonl'} does not exist.")
            tables = []
        else:
            with open(results_path / "tables.jsonl", "r") as f:
                print("Loading existing tables from", results_path / "tables.jsonl")
                lines = f.readlines()
                if len(lines) == 0:
                    print("No tables found in", results_path / "tables.jsonl")
                    tables = []
                else:
                    tables = [json.loads(line) for line in lines] 
    return tables

def load_experiment_setting(args: DictConfig):
    results_path = Path(args.results_path)
    with open(results_path / "experiment_setting.json", "r") as f:
        return json.load(f)
    
def save_experiment_setting(args: DictConfig, experiment_setting: dict):
    results_path = Path(args.results_path)
    with open(results_path / "experiment_setting.json", "w") as f:
        json.dump(experiment_setting, f)

def save_outputs(args: DictConfig, tab_id, tables: list, metrics: dict):
    # save the tables - think about the format here - it might actually be good to save these as a
    # single, large pandas dataframe with an additional key representing the table/group of papers being
    # compared. For now, this is a jsonl file with one line per table/group of papers.
    # results_path = Path(args.results_path)
    # f"results/{args.data._id}/{tab_id}/{args.endtoend.difficulty}/{args.endtoend.model_type}/{args.mode}/{args.endtoend.retry_type}"
    results_path = f'results/{args.data._id}/{tab_id}/{Path(args.results_path).split("/")[2:].join("/")}'
    
    with open(results_path / "tables.jsonl", "w") as f:
        for table in tables:
            # f.write(json.dumps({"id": table["id"], "tabid": table["tabid"], "table": table["table"], "caption": table["caption"], **({"type": table["type"]} if hasattr(table, 'type') else {})}) + "\n")
            # f.write(json.dumps({"id": table["id"], "tabid": table["tabid"], "table": table["table"], "caption": table["caption"], "type": table["type"]}))
            f.write(json.dumps(table) + "\n")
        print("\nsaveing outputs in utils.py ")
    with open(results_path / "metrics.json", "w") as f:
        json.dump(metrics, f)
    print("Saving outputs:")
    # print("Tables:", tables)
    print("Metrics:", metrics)

def load_total_outputs(args: DictConfig):
    # load the tables: if the tables are stored in a jsonl file, then we can just read them in
    results_path = Path(args.total_results_path)
    print(results_path)
    if not results_path.exists():
        print(f"Results path {results_path} does not exist.")
        
    else:
        if not (results_path / "total_table.json").exists():
            print(f"Tables file {results_path / 'tables.jsonl'} does not exist.")
            total_data = {"gold": [], 
                          "pred": {"hard": {"baseline": {"mistral": [], "gpt4":[]}, 
                                                "ours": {"mistral": [],  "gpt4":[]}}, 
                                    "medium": {"baseline": {"mistral": [], "gpt4":[]}, 
                                                "ours": {"mistral": [],  "gpt4":[]}}, 
                                    "easiest": {}}}
        else:
            with open(results_path / "total_table.json", "r") as f:
                print("Loading existing tables from", results_path / "total_table.json")
                total_data = json.load(f)
    return total_data
 
def save_total_outputs(args: DictConfig, tables: list, metrics: dict):
    # save the tables - think about the format here - it might actually be good to save these as a
    # single, large pandas dataframe with an additional key representing the table/group of papers being
    # compared. For now, this is a jsonl file with one line per table/group of papers.
    results_path = Path(args.results_path)
    with open(results_path / "tables.jsonl", "w") as f:
        for table in tables:
            f.write(json.dumps({"tab_id": table.tabid, "table": table.values, "caption": table.caption, **({"type": table.type} if hasattr(table, 'type') else {})}) + "\n")

    with open(results_path / "metrics.json", "w") as f:
        json.dump(metrics, f)
    print("Saving outputs:")
    # print("Tables:", tables)
    print("Metrics:", metrics)
