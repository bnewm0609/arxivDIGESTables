from datetime import datetime
import os
import sys
import random
import json
import traceback
from typing import Any, Dict, List
from concurrent.futures import ProcessPoolExecutor

import hydra
from omegaconf import DictConfig, OmegaConf

from paper_comparison.metrics import load_metrics
from paper_comparison.data import load_data
from paper_comparison.endtoend import load_endtoend
from paper_comparison.utils import save_outputs, get_results_path, convert_paths_to_absolute, load_outputs, load_experiment_setting, save_experiment_setting

def initialze_experiment(args: DictConfig):
    args.results_path = get_results_path(args)
    print(args.results_path)
    args.timestamp = str(datetime.now())
    
    ## setup seeds
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)

    convert_paths_to_absolute(args)

    if args.notes is not None:
        args.results_path += f"_note-{args.notes}"

    os.makedirs(args.results_path, exist_ok=True)

    # save the command that was run
    with open(os.path.join(args.results_path, "commands.txt"), "a") as f:
        f.write(" ".join(sys.argv) + "\n")

def process_paper(
    args: DictConfig,
    index: int,
    paper: Dict[str, Any],
    endtoend: Any,
    retry_num: int
):
    """Process a single paper using the endtoend pipeline and save the results.
    """
    tab_id = paper['y'].tabid
    print(f"[START] tab_id: {tab_id}")
    try:
        # Use gold_caption if the difficulty is medium
        gold_caption = None
        if args.difficulty == "caption":
            gold_caption = paper['y'].caption
        
        # Save path for all experiments in this setting
        save_path = f"{args.results_path}/tmp/{args.difficulty}/{tab_id}/{args.endtoend.model_type}/{args.endtoend.name}"
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        # Error path for all experiments in this setting
        error_path = f"{args.results_path}/tmp/errors/{args.endtoend.model_type}/{args.endtoend.name}"
        if not os.path.exists(error_path):
            os.makedirs(error_path, exist_ok=True)

        # For each try/experiment, run the endtoend pipeline
        for r_id in range(retry_num):
            start_time = datetime.now()

            error_data = None
            # If this tab_id, r_id pair is already processed, skip
            if os.path.exists(f"{save_path}/try_{r_id}.json"):
                print(f"\tretry_id: {r_id} already exists")
                continue
            elif os.path.exists(f"{error_path}/{tab_id}___{r_id}.json"):
                with open(f"{error_path}/{tab_id}___{r_id}.json", "r") as f:
                    error_data = json.load(f)

            column_num = len(paper['y'].schema)
            table_set = endtoend(args=args, sample=paper, tab_id=tab_id, index=index, column_num=column_num, gold_caption=gold_caption)

            # If there is an error in the output, save the error file and continue
            is_error = False
            for table in table_set:
                if "text" in table:
                    is_error = True
                    break
            
            if not is_error:   
                # Remove error file if it was processed successfully
                if error_data is not None:
                    for table in table_set:
                        table["error_counts"]["over_max_length_error"] = True
                        table["error_counts"]["have_length_error"] = True
                        table["error_counts"]["restarted_count"] = error_data["restarted_count"] + 1
                    os.remove(f"{error_path}/{tab_id}___{r_id}.json")
                with open(f"{save_path}/try_{r_id}.json", "w") as f:
                    json.dump(table_set, f)
                print(f"\tResults saved to: {save_path}/try_{r_id}.json")
            else:
                print(f"\tError in tab_id: {tab_id} / retry_id: {r_id}")
                # Save error file
                restarted_count = 0
                if error_data is not None:
                    restarted_count = error_data["restarted_count"] + 1
                with open(f"{error_path}/{tab_id}___{r_id}.json", "w") as f:
                    json.dump({ "tab_id": tab_id, "r_id": r_id, "restarted_count": restarted_count }, f)

            print(f"\tretry_id: {r_id} / time taken (s): {(datetime.now() - start_time).total_seconds()}")
            break
        print(f"[COMPLETE] tab_id: {tab_id}")
        return True, tab_id, None
    except Exception as e:
        print(f"[ERROR] tab_id: {tab_id} ({e})")
        print(traceback.format_exc())
        return False, tab_id, traceback.format_exc()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(args: DictConfig) -> None:
    """Entrypoint for running experiments.

    Registers resolvers, and sets up the experiment runner based on the passed configuration arguments.

    Args:
        args (DictConfig): The experiment configuration arguments.
    """
    OmegaConf.register_new_resolver(
        "extract_path",
        lambda path, num_segments: "-".join(os.path.splitext(path)[0].strip("/").split("/")[-num_segments:]),
    )
    print(args)
    # assert args.endtoend is not None or (args.schematizer is not None and args.populator is not None)
    args.mode = "baseline" if args.endtoend.name == "baseline_outputs" else "ours"
    
    initialze_experiment(args)
    data = load_data(args)
    evaluator = load_metrics(args) 
    print(f"Running endtoend {args.endtoend.name}")
    endtoend = load_endtoend(args)
    print(f"there are {len(data)} number of papers\n")
    retry_num = 1
    # tables = load_outputs(args)

    # Process batch_size papers in parallel
    batch_size = 1
    futures = []
    # data_name = "metric_validation_0"
    # directory_path = f'results/{data_name}'
    # load json file

    # process a single paper for debugging
    # process_paper(args, 0, data[0], endtoend, retry_num)
    
    with ProcessPoolExecutor(max_workers=batch_size) as executor:
        for index, paper in enumerate(data):
            future = executor.submit(
                process_paper, 
                args,
                index,
                paper,
                endtoend,
                retry_num
            )
            futures.append(future)
    
    
    errors = []
    for future in futures:
        result, tab_id, error = future.result()
        # Save non-code related errors
        if not result:
            # get full error message (file name, line number, etc.)
            errors.append({ "tab_id": tab_id, "error": error })
            # save with current timestamp
            try:
                with open(f"{args.results_path}/errors/log_{args.timestamp}.json", "w") as f:
                    json.dump(errors, f)
            except FileNotFoundError:
                with open(f"{args.results_path}/tmp/errors/log_{args.timestamp}.json", "w") as f:
                    json.dump(errors, f)
    
    if len(errors) > 0:
        print(f"Errors occurred during processing. Check the error log in {args.results_path}/errors/log_{args.timestamp}.json")

if __name__ == "__main__":
    main()