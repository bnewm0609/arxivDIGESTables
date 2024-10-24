# Creating ArxivDIGESTables


## High quality dataset
Download the files https://huggingface.co/datasets/blnewman/SciTabNet

There are three relevant files:
- `tables.jsonl`: a list of tables. Each line is a `json` object with the following fields:
    - `tabid`: a unique hash for the table that comes from the latex-processing pipeline
    - `table`: a python dictionary dump of the table created from`pd.DataFrame.to_dict()`. Can be loaded back into pandas with `pd.DataFrame(table['table'])`. The index of the data frame is the the `corpus_id` of the full text of the paper (which can be accessed at `data/full_texts/{corpus_id}.jsonl`).
    - `row_bib_map`: a list of `json` objects, where each represents a citation from the table. (This shouldn't be needed anymore, but I'm leaving it in in case it's useful for now. Will probably be deleted soon.) Each object contains:
        - the `corpus_id` of the full text of the paper
        - the `row` in the table that `corpus_id` corresponds to
        - the `type` of the citation - `"ref"` means it's an external reference and `"ours"` means that the paper containing the table is represented in the given row.
        - the `bib_hash_or_arxiv_id` associated with that corpus_id. `arxiv_id` is used when the `type` is `"ours"` and the `bib_hash` is used when it's `"ref"`. The `bib_hash` comes from the latex-processing pipeline and it's a function of the citation text and the citing paper's arxiv_id.
        - `caption`: the table's caption
        - `in_text_ref`: a list of paragraphs where the table is refered to in the main text
        - `arxiv_id`: the arxiv id of the paper that table comes from
- `papers.jsonl`: a list of all the papers needed for generating the tables. Each paper entry contains:
    - `tabids`: a list of ids for the tables the paper is cited in.
    - `corpus_id`, the corpus id of a paper. If the full text is available, it can be accessed at `data/full_texts/{corpus_id}.jsonl`. (For the small subset, all of these texts should be available.)
    - `title`: the title of the paper from s2.
    - `paper_id`: the s2 id of the paper.
- `full_text.jsonl.gz`: contains the full texts for the papers in `papers.jsonl`


## Creating ArxivDIGESTables

### Data Source
We recommend following the directions at the following fork of (unarxiv)[https://github.com/bnewm0609/unarXive/tree/master/src] to download the latex for the submitted papers. When creating the dataset, we used an internal data source and batched the papers by submission month. For the purposes of these instructions, put each month's worth of papers in a single `.tar` file a directory named `in_tar/`.

### Extracting Tables
Then clone the fork of the (unarxiv)[https://github.com/bnewm0609/unarXive/tree/master/src] repo and run steps 1 and 3 under the heading `Usage`. Skip step 2, because we use the S2 apis to do the citation matching. Step 3 looks like this. (`arxiv-metadata-oai-snapshot.sqlite` is created by step 1). In this example, the extracted tables are saved in a file named `out_xml`.

```
python unarXive/src/prepare.py in_tar/ out_xml/ arxiv-metadata-oai-snapshot.sqlite
```

### Filtering Tables and Matching Rows to Papers
The first round of filtering the tables and saves them to a pandas-compatible json format. In this example, the tables are saved in a file named `out_xml_filtered`
```
python scripts/data_processing/extract_tables.py out_xml/<yymm>.jsonl out_xml_filtered/<yymm>_dataset.jsonl
```

### Obtaining Table Citation Metadata
Match the citations to the Semantic Scholar database. (Note: some of these require calling semantic-scholar internal apis, specifically for parsing titles and getting corpus ids from titles, or require a semantic scholar API key. Email the authors for more information.) In this example, bibliography metadata is stored in `out_bib_entries.jsonl`.

```
python scripts/data_processing/populate_bib_entries.py out_xml/<yymm>.jsonl out_xml_filtered/<yymm>_dataset.jsonl out_bib_entries.jsonl
```

### Downloading full texts
Downloads the full texts of the cited papers (when available.) This uses an api that needs authentication. Contact the authors for more information.

```
python scripts/data_processing/download_full_texts.py data/arxiv_tables/<yymm>_full_texts.jsonl
```

### Final clean-up
This creates the `tables.jsonl` and `papers.jsonl` files.
```
python scripts/data_processing/create_tables_and_papers_datasets.py out_xml_filtered/<yymm>_dataset.jsonl tables.jsonl papers.jsonl
```

### Summarize dataset
This prints a summary of the contents of `tables.jsonl`.
```
python scripts/data_processing/summarize_dataset.py tables.jsonl
```

## Manually Editing Tables
Some of the tables had parsing bugs and needed to be manually corrected. The following script allows for editing and saving individual tables.
```
!pip install dtale
python scripts/data_processing/data_editor.py data/arxiv_tables/dataset.jsonl <table_id> --out_file data/dataset.jsonl
```