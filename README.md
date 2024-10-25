# ArxivDIGESTables: Synthesizing Scientific Literature into Tables using Language Models.


## How do I access the data?
The data is available on [huggingface](https://huggingface.co/datasets/blnewman/arxivDIGESTables). The tables are stored in a format that does not play nice with huggingface's default dataset builder. Rather than using `load_dataset("blnewman/arxivDIGESTables
)`, you should download the data directly from: [huggingface.co/datasets/blnewman/arxivDIGESTables](https://huggingface.co/datasets/blnewman/arxivDIGESTables/tree/main).

For the high quality data, you should download `papers.jsonl`, `tables.jsonl`, and `full_texts.jsonl.gz`. 
If you want more tables that are less stringently filtered and do not have associated full texts, you can download `papers_medium_quality.jsonl` and `tables_medium_quality.jsonl`.
- The `papers*.jsonl` files include information about the papers cited in the tables including their S2 corpus ids, title, abstract and the ids of what tables they can be found in. These are mostly useful for analysis as almost all of the information is also included in the `tables*.jsonl` files.
- The `tables*.jsonl` files include:
    - `tabid`: an id for each table
    - `table`: the table itself, which is a nested json dictionary
    - `row_bib_map`: which maps each row of the table to the corpus id, title, and abstract for the paper cited in that row.
    - `caption`: the table's caption
    - `in_text_ref`: a list of paragraphs where the table is refered to in the main text
    - `arxiv_id`: the arxiv id of the paper that table comes from
- `full_text.jsonl.gz`contains the full texts for the papers in `papers.jsonl`.

For information on curating the dataset, see the `data` directory

## How do I generate predictions?
Information for how we generated tables is in the `experiment` directory.
If you want to see our raw predictions, they are available in `predictions/predictions.jsonl`.

If you're running your own pipeline to generate tables, you'll want to make sure your predicted tables are formatted like those in `predictions/predictions.jsonl`. Each line is a json object with the keys `"table"` and `"metadata"`. `metadata` can be used to store any relevant information about the tables were generated. The only required key is `"tabid"`, whose value is the id of the table that's been generated.
Example:
```
{"table": {...}, "metadata": {"tabid": "bb09b7e1-2ab7-4193-922a-1b1b93486e83", ...}}
```

## How do I run the evaluation metric?
To run `DecontextAlign` on generated tables:

First, install the requirements `pip install -r requirements.txt`.
If necessary, add your together.ai API key to your environment. (e.g. `os.environ["TOGETHER_API_KEY"] = ...`)

Then, run `metric/run_eval.py`.

```
python metric/run_eval.py --gold_tables <path_to_gold_tables> --pred_tables <path_to_predicted_tables> --out_file <out_file> --featurizer <featurizer> --scorer <scorer> --threshold <threshold>
```

The options for `featurizer` are:
- `"name"`: just uses the column name
- `"values"`: uses the column name and values in the column
- `"decontext"`: queries Mixtral to add additional context to the column name based on the table. NOTE: you need a [together.ai](https://www.together.ai/) api key to prompt Mixtral for the decontextualization

The options for `scorer` are:
- `"exact_match"`
- `"jaccard"`
- `"sentence_transformers"`
- `"llama3"`: Note: also requires a together.ai api key

The parameter `threshold` is a float between 0 and 1 that indicates the minimum proportion of overlap/similarity there must be for a predicted and gold column header to count as a match.

Our best metric was `"decontext"`, `"sentence_transformers"`, `0.7`. E.g.:
```
python metric/run_eval.py --gold_tables hf_dataset/tables.jsonl --pred_tables predictions/predictions.jsonl --out_file predictions/predictions_decontext-st-0.7.jsonl --featurizer decontext --scorer sentence_transformers --threshold 0.7
```

The contents of `<out_file>` will be identical to the passed `<path_to_predicted_tables>`, except each json object will have an additional key `"scores"`, which contains the `recall` (the proportion of matched columns), any `alignment` that was produced along with the score for that alignment, and the passed `featurizer`, `scorer` and `threshold`, all of which can help with organizing results. See `predictions/predictions_decontext_with_scores.jsonl` for an example.