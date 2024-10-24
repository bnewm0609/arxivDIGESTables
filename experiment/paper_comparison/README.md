# Table Generation Pipeline

## Generation

Navigate to the `configs/config.yaml`. Here, you can choose between `endtoend` options by selecting either `ours_outputs` or `baseline_outputs`.

If you wish to change the model type or the number of tables you want to generate per try, you can do so by modifying the `model_type` (mixtral, gpt3.5, and gpt4 are available) and `num_commonality` parameter in either the `configs/endtoend/baseline_outputs.yaml` or `configs/endtoend/ours_outputs.yaml`. (Other hyperparameters are fixed in current experiment setup.)

If you want to change the input data, you can do so by modifying the `path` parameter in the `configs/data/abstracts.yaml`.

To generate tables using the paper sets input data at the `path` above: 
```
python paper_comparison/run_batch.py
```
After running, the results of each try and the path will be output to the terminal. This path contains following items:
 - `results/{args.data._id}/{table_id}/{args.endtoend.model_type}/{args.endtoend}`: `retry` number of files named with `try_{retry_idx}.json` containing `num_commonality` number of tables.
 - `commands.txt`: a list of commands that were run to obtain the results in the directory. Most recent at the bottom.

## Code Structure
- `paper_comparison/run_batch.py`: Run experiment with total input data and save predicted tables
- `paper_comparison/endtoend.py`: Transform collections of papers into tables (with baseline prompting and our prompting)
- `paper_comparison/generation.py`: Various generation utilities
- `data/prompt_ver3.json`: Prompts used in current experiment setup (Past prompts are in `data/prompt_ver2.json`)

## Post Analysis

 To determine the number of tables that need to be excluded from the experiment: 
 ```
python paper_comparison/check_experiment.py
```
 After running, the tab_ids of tables to be excluded will be saved in file named `00_removed_tabids.json`. These tables are cases where the number of calls exceeds 5 to generate them.