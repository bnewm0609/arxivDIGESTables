from argparse import ArgumentParser
from collections import defaultdict
import copy
import functools
import gzip
import json
import multiprocessing
import os
import re
import sys
from typing import List
import warnings

from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

from summarize_dataset import get_aspect_type


def soupify(table_json):
    soup = BeautifulSoup(table_json, "lxml-xml")
    for row in soup.find_all("row"):
        row.name = "tr"
    for cell in soup.find_all("cell"):
        cell.name = "td"
    for tex_math in soup.find_all("texmath"):
        # remove the "texmath"
        tex_math.extract()
    return soup


# Define our filters
def has_x(table_soup):
    return "✗" in " ".join(table_soup.strings)


def not_too_long_15e3(table_soup):
    return len(str(table_soup)) < 15e3


def not_too_long_5e3(table_soup):
    return len(str(table_soup)) < 5e3


def not_too_long_or_short(table_soup):
    # return len(str(table_soup)) < 5e3
    return 398 < len(str(table_soup)) < 15e3


def has_rows(table_soup):
    return table_soup.find("tr")


def has_max_2_sub_tables(table_soup):
    return len(table_soup.find_all("table")) <= 2


def has_at_least_2_cols(table_soup):
    # td is number of cells, so use combo of # cells and # of rows to get columns
    return len(table_soup.find_all("td")) >= 4 and len(table_soup.find_all("tr")) >= 2


def has_at_least_2_rows(table_soup):
    return len(table_soup.find_all("tr")) >= 2


def has_cites(table_soup):
    soup_text = " ".join(table_soup.strings)
    return len(table_soup.find_all("cit")) > 0 or ("et al" in soup_text)


def has_at_least_2_cites(table_soup):
    return len(table_soup.find_all("cit")) >= 2


def has_cites_in_first_row_or_col(table_soup):
    """
    Checks for citations in the first row or column
    (this is quite restrictive)
    """
    min_num_cites = 2
    trs = table_soup.find_all("tr")

    # check the first, non-empty row
    i = 0
    while i < len(trs):
        first_row = trs[i]
        cells = first_row.find_all("td")

        # skip any all-empty rows
        if all([not cell.text.strip() for cell in cells]):
            i += 1
            continue

        if len(first_row.find_all("cit")) >= min_num_cites:
            return True
        else:
            break

    # check the first column (usually not empty)
    # first_col_citations = [row.find("cit") for row in trs if row.find("cit") is not None]
    first_col_citations = [row.find_all("td")[0].find("cit") for row in trs]
    return len([cell for cell in first_col_citations if cell is not None]) >= min_num_cites


def has_cites_in_rows_or_cols(table_soup):
    """
    Checks *any* row or col to see if it has >2 cites
    """
    min_num_cites = 2
    trs = table_soup.find_all("tr")

    valid_table = False
    max_num_cells = 0
    for row in trs:
        cells = row.find_all("td")
        # skip any all-empty rows
        if all([not cell.text.strip() for cell in cells]):
            continue

        # track max number of cells per row to help with processing column
        max_num_cells = max(max_num_cells, len(cells))

        if len(row.find_all("cit")) >= min_num_cites:
            valid_table = True
            break

    # check columns. Assumes that we don't have weird multicolumn stuff going on,
    # so it's a bit coarse. E.g. row[0][3] is the same column as row[3][3]
    if not valid_table and max_num_cells > 0:
        for col_i in range(max_num_cells):
            col_citations = []
            for row in trs:
                try:
                    col_citations.append(row.find_all("td")[col_i].find("cit"))
                except IndexError:
                    continue
            # print(col_citations)
            if len([cell for cell in col_citations if cell is not None]) >= min_num_cites:
                valid_table = True
                break
    return valid_table


def has_max_one_cite_per_cell(table_soup):
    """
    We don't want any tables with more than one citation per cell.
    Returns True if any cell has a maximum of one citation
    """
    max_cites_per_cell = 1
    trs = table_soup.find_all("tr")

    valid_table = False
    for row in trs:
        for cell in row.find_all("td"):
            if len(cell.find_all("cit")) > max_cites_per_cell:
                return False
    return True


# FLOAT_REGEX = re.compile("\d\.\d")
FLOAT_REGEX = re.compile("\.\d")


def has_no_floats(table_soup):
    for cell in table_soup.find_all("td"):
        cell_text = cell.get_text()
        if cell_text and FLOAT_REGEX.search(cell_text) is not None:
            return False

    # some tables have floats in paragraphs, which is confusing but
    # not always recovered for some reason
    for cell in table_soup.find_all("p"):
        cell_text = cell.get_text()
        if cell_text and FLOAT_REGEX.search(cell_text) is not None:
            return False
    # for s in table_soup.strings:
    #     s = s.strip()
    #     if s and FLOAT_REGEX.search(s) is not None:
    #         return False
    return True


def has_table_cells(table_soup):
    return table_soup.find("td") is not None


def has_no_figures(table_soup):
    for cell in table_soup.find_all("td"):
        cell_text = cell.get_text()
        if cell_text and r"{{figure" in cell_text:
            return False

    for cell in table_soup.find_all("p"):
        cell_text = cell.get_text()
        if cell_text and r"{{figure" in cell_text:
            return False
    return True


DEFAULT_TABLE_LABELS = [
    not_too_long_15e3,
    not_too_long_or_short,
    has_table_cells,
    has_at_least_2_cites,
    has_max_2_sub_tables,
    has_at_least_2_cols,
    has_at_least_2_rows,
    has_no_floats,
    has_no_figures,
    has_cites_in_rows_or_cols,
    has_cites_in_first_row_or_col,
    has_max_one_cite_per_cell,
]

DEFAULT_TABLE_FILTERS = [
    has_cites_in_rows_or_cols,
    has_at_least_2_cites,
    # has_no_floats,
    not_too_long_or_short,
    has_at_least_2_cols,
    has_max_2_sub_tables,
    has_at_least_2_rows,
    # has_no_figures,
    has_table_cells,
]


# this list of colors is not complete. Many colors need to be manually removed.
COLORS = r"((alice)?blue|black|(mid)?gr[ae]y|red|(dark)?green|tablewhite|tableblue)"
COLORS_RE = rf"{COLORS}(\!\d\d?|(?=[✗✓]))"


def is_na(text):
    return text.lower() == "n/a" or not text.strip() or text == "\u2216"


def extract_valid_tables(path, table_filters, label_tables=False):
    """
    If `label_tables` is True, return all of the tables, but list
    what labels are true/false for each of them.


    path by default is "arxiv_dump/out_xml/2310.00000-07773.jsonl"

    This is the output of running the following on the s2 cluster:

    1. Download the gzipped latex data into `in_latex_s3/{month}`:
    `aws s3 cp --recursive "s3://ai2-s2-scholarphi-pipeline-prod/daq/arxiv-source-data/bymonth/2310" in_latex_s3/2310`
    or
    `aws s3 cp --recursive "s3://ai2-s2-scholarphi-pipeline-prod/daq/arxiv-source-data/bymonth/2309" in_latex_s3/2309`

    The directory that's created has a bunch of .gz files in it - one with all the latex for that submission

    2. Bundle the .gz file into a tar file
    `tar cvf in_tar/2310.00000-07773.tar in_latex_s3/2310/`


    3. Extract the xml from the latex files
    `python unarXive/src/prepare.py in_tar/ out_xml/ arxiv-metadata-oai-snapshot.sqlite`

    This takes all of the latex that's packaged in `in_tar` and outputs its associated xml in `out_xml`

    Then on your local machine
    ```
    scp benjaminn@s2-cirrascale-10.reviz.ai2.in:~/nfs/arxiv_tables/out_xml/2310.00000-07773.jsonl arxiv_dump/out_xml/
    ```

    Then run this script:

    Then run `populate_bib_entries"
    Finally, run
    python scripts/data_processing/download_full_texts.py data/arxiv_tables/2308_papers.jsonl
    """
    valid_tables = []
    if os.path.splitext(path)[1] == ".gz":
        f = gzip.open(path, "r")
    else:
        f = open(path)
    with f:
        added_tables = set()
        for line in tqdm(f):
            paper = json.loads(line)
            filtered_tables = {}
            for key, table in paper["tables"].items():
                if table["table"]:
                    table_soup = soupify(table["table"])

                    if label_tables:
                        # label the tables with the filters they pass
                        labels = {}
                        for flter in table_filters:
                            labels[flter.__name__] = flter(table_soup)
                        labels["len"] = len(str(table_soup))
                    else:
                        # Filter tables
                        exit_early = False
                        for flter in table_filters:
                            if not flter(table_soup):
                                # exit early as soon as a filter is wrong
                                exit_early = True
                                break

                        if exit_early:
                            continue

                    # Keep the outermost table always. But prevent adding smaller tables
                    # Remove duplicates (as long as the larger table comes first, the smaller ones
                    # won't make it in). Usually the larger seems to come first.
                    if table_soup.find("table") in added_tables:
                        continue
                    else:
                        for sub_table in table_soup.find_all("table"):
                            added_tables.add(sub_table)

                    filtered_tables[key] = table
                    filtered_tables[key]["soup"] = table_soup
                    if label_tables:
                        filtered_tables[key]["labels"] = labels

            if filtered_tables:
                new_paper = {k: v for k, v in paper.items() if k != "tables"}
                new_paper["tables"] = filtered_tables
                valid_tables.append(new_paper)

            # For debugging
            # if len(valid_tables) > 10:
            #     break

    return valid_tables


def split_references_column(table_df):
    # next, break the citation into their own column with the heading "References"
    # this new References column will be the *index* of the dataframe, so all it's elements
    # must be unique
    # determine which column has the most references

    # use iloc in case the name of the column is repeated
    column_with_cites = table_df.iloc[:, 0]  # by default it's the first one
    max_num_cites = -1
    for col_i, _ in enumerate(table_df.columns):
        num_cites = 0
        for cell_val in table_df.iloc[:, col_i]:
            matches = re.search("{{cite:[a-f\d]{7}}}", cell_val)
            if matches is not None:
                num_cites += 1
        if num_cites > max_num_cites:
            max_num_cites = num_cites
            column_with_cites = table_df.iloc[:, col_i]

    if isinstance(column_with_cites, pd.DataFrame):
        column_with_cites = column_with_cites.agg("".join, axis=1)
    references_col = []

    new_column_without_cites_name = column_with_cites.name

    new_column_without_cites = []
    no_cite_count = 0
    for cell_val in column_with_cites:
        matches = re.search("{{cite:[a-f\d]{7}}}", cell_val)
        if matches is None:
            references_col.append(f"no_cite-{no_cite_count}")
            new_column_without_cites.append(cell_val)
            no_cite_count += 1
        else:
            references_col.append(matches[0])
            new_cell_val = cell_val.replace(matches[0], "")
            if not new_cell_val:
                new_cell_val = "-"
            new_column_without_cites.append(new_cell_val.strip())

    if any([val != "-" for val in new_column_without_cites]):
        if new_column_without_cites_name == "References":
            # This only gets triggered in weird cases where one of the columns is "References" but
            # we don't successfully parse out all of the citations (eg if there is more than one cite
            # in the column)
            table_df = table_df.rename(columns={new_column_without_cites_name: "References_OLD"})
            table_df["References_OLD"] = new_column_without_cites
            new_column_without_cites_name = "References_OLD"
        else:
            table_df[new_column_without_cites_name] = new_column_without_cites
        try:
            table_df.insert(0, "References", references_col)
        except ValueError:
            print("Error inserting into table_df")
            raise ValueError
    else:
        # if the column just has citations, rename the column
        table_df = table_df.rename(columns={new_column_without_cites_name: "References"})
        # and reorder them so that "References" is first
        reordered_cols = ["References"] + [col for col in table_df.columns if col != "References"]
        table_df = table_df[reordered_cols]
    assert table_df.columns[0] == "References"
    # if table_df.columns[0] != "References":
    #     breakpoint()
    return table_df, new_column_without_cites_name


def postprocess_table_df(table_df):
    """
    Converts a list, where each element is row, into a dictionary representing
    the table. This conversion is done using pandas and then a large amount of
    post-processing this conversion, this method
    """

    # if the citations are in the columns, then change them to the rows
    if " ".join(table_df.columns).count("{{cite:") > 0:
        original_col_0 = table_df.columns[0]
        table_df = table_df.set_index(original_col_0)
        table_df = table_df.transpose()
        try:
            table_df = table_df.reset_index(names=original_col_0)
        except ValueError:
            # something went wrong so... transpose back
            table_df = table_df.transpose().reset_index()
            # breakpoint()

    def process_cell(cell):
        """The heuristics defined here are a start for normalizing cell content. In reality they
        are not sufficient. There is a lot of manual post-editing that must be conducted on the
        tables."""
        # replace non-breaking space with normal space
        if isinstance(cell, tuple):
            cell = str(cell)
        cell = cell.replace("\u00a0", " ")
        cell = cell.strip()

        # normalize apostrophes
        cell = re.sub("[’]", "'", cell)
        # binary no
        if cell == "X":
            cell = "\u2717"

        if cell == "tablered":
            cell = "no"
        # cell = re.sub(f"{COLORS}\u2717", "\u2717", cell)
        # binary yes - standardize
        cell = re.sub("\u2714", "\u2713", cell)
        if cell == "tablegreen":
            cell = "yes"

        # remove color annotations from end
        cell = re.sub(f"{COLORS_RE}", "", cell)
        # and from beginning - this is potentially dangerous...
        cell = re.sub(rf"^{COLORS}", "", cell)

        # remove latex positioning markers that can come in:
        cell = re.sub(r"\[[cl]\]", "", cell)

        # remove font size info
        cell = re.sub(r"^\d\d+em", "", cell)

        # empty cells should be "-" instead
        cell = re.sub(r"(N/A|none)", "-", cell)
        cell = cell.strip()
        if cell == "":
            cell = "-"
        return cell

    table_df = table_df.map(process_cell)
    table_df.columns = table_df.columns.map(process_cell)
    # print(table_df.columns)

    # if the cells have citations and other information, put the citations into a new cell
    table_df, old_col_with_cites_name = split_references_column(table_df)
    # TODO: if rows have the same reference, then combine their values into a list
    return table_df, old_col_with_cites_name


def merge_rows(table, row_i, row_j):
    new_row = []
    for val_i, val_j in zip(table[row_i], table[row_j]):
        if val_i == val_j or not val_j.strip():
            new_row.append(val_i)
        elif not val_i.strip():
            new_row.append(val_j)
        else:
            new_row.append("-".join([val_i, val_j]))
    table[row_i] = new_row
    del table[row_j]
    return table


def soup_to_json(table_soup, verbose=False):
    # first, determine the number of columns as the max number of cells in a row
    num_cols = max(
        [len(row.find_all("td")) for row in table_soup.find_all("tr")]
        + [
            sum([int(cell.attrs.get("cols", "1")) for cell in row.find_all("td")])
            for row in table_soup.find_all("tr")
        ]
    )
    # next, determine the number of rows:
    num_rows = len(table_soup.find_all("tr"))

    if verbose:
        print(num_rows, num_cols)

    # next, extract the values. Some rows are "header" rows and contain explanatory info.
    # The rows we are unable to parse are tracked separately in a "incomplete_rows" field

    table = {
        "incomplete_rows": [],
        "table": [["" for ci in range(num_cols)] for ri in range(num_rows)],
        "old_citation_column": None,
    }
    columns = [[] for _ in range(num_cols)]
    transposed_table = False

    # Next, fill in table[row_i][col_i]
    header_rows = []
    seen_cites = False
    for row_i, row in enumerate(table_soup.find_all("tr")):
        cells = row.find_all("td")

        col_i = 0
        # if a row does not contain any citations and we only have one row in the table, then say it's part of the header row
        num_cites = len([cell for cell in cells if cell.find("cit") is not None])
        if num_cites > 0:
            seen_cites = True
        if num_cites == 0 and row_i <= 1 and not seen_cites:
            # This shouldn't always be a header... eg if "ours" is first
            if verbose:
                print(f"no cites in row: {row_i}, adding as header")
            header_rows.append(row_i)

        # determine if the current row is a header row
        if len(cells) < num_cols and not seen_cites:
            if verbose:
                print(f"not enough cols in row: {row_i}, adding as header")
            header_rows.append(row_i)
        elif len(cells) < num_cols or any(["{{figure:" in cell.text for cell in cells]):
            # for cell in cells:
            table["incomplete_rows"].append(
                {
                    "row_idx": row_i,
                    "cells": [cell.text for cell in cells],
                }
            )
            continue

        for cell in cells:
            # acount for multi-column cells
            num_spanning_cols = int(cell.attrs.get("cols", "1"))
            for col_offset in range(num_spanning_cols):
                cell_text = cell.text
                # account for multi-row cells
                multirow_cell = re.search(r"(\d)\*(.+)", cell_text)
                if multirow_cell:
                    count = int(multirow_cell[1])
                    cell_text = multirow_cell[2]
                    if verbose:
                        print(count, cell_text)
                    for row_offset in range(1, count):
                        # add this to subsequent rows
                        try:
                            table["table"][row_i + row_offset][col_i + col_offset] += cell_text.strip()
                        except IndexError:
                            print(
                                "Index error assigning value. This probably means we were unable to parse the table correctly. Skipping..."
                            )
                            return {
                                "table": table["table"],
                                "incomplete_rows": table["incomplete_rows"],
                                "table_dict": {},
                            }

                # if a cell contains a horizontal line, don't add it but mark it as a header row
                if re.search("\(r\)\d-\d", cell_text):  #  and not seen_cites:
                    # print("horizontal line added as header_row")
                    # header_rows.append(row_i + 1)
                    cell_text = re.sub("\(r\)\d-\d", "", cell_text)
                    # continue

                if (
                    is_na(cell_text.strip())
                    and not table["table"][row_i][col_i + col_offset]
                    and not row_i in header_rows
                ):
                    cell_text = "-"

                if not table["table"][row_i][col_i + col_offset]:
                    table["table"][row_i][col_i + col_offset] += cell_text.strip()
            col_i += num_spanning_cols

    # after we're done filling in the table, collapse the header row
    # merge the first two rows until we reach the header row
    if verbose:
        print(header_rows)
    if header_rows:
        for _ in range(max(header_rows)):
            table["table"] = merge_rows(table["table"], 0, 1)

    # remove any empty rows
    table_filtered = []
    for row in table["table"]:
        if any(row):
            table_filtered.append(row)
    table["table"] = table_filtered

    if verbose:
        for row in table["table"]:
            print(row)

    # next, assume the first row has the column headers and the first col has the row headers
    if not any(["".join(row) for row in table["table"]]):
        table_dict = {}
    else:
        table_df = pd.DataFrame(table["table"][1:], columns=table["table"][0])
        table_df, old_col_with_cites_name = postprocess_table_df(table_df)
        table["old_citation_column"] = old_col_with_cites_name
        table_dict = table_df.to_dict(orient="list")

    table["table_dict"] = table_dict
    return table


def get_table_row_bib_map(table_json, bib_hashes, paper_id) -> List:
    """
    Uses the heuristic that if a table contains a row that doesn't have a citation,
    then that row represents the containing paper, as long as the cell doesn't contain
    certain words e.g. "standard".

    Returns a List where each element represents a row of the table. Each element contains
    a row number, the corpus id, bib_hash or arxiv id, and whether the row is the paper
    with the table ("ours") or an external reference ("ref").

    TODO: could also be "above". there could also be more than one "ours" row.
    """

    table_row_bib_map = []
    cite_id_map = {bib_ref[:7]: bib_ref for bib_ref in bib_hashes if bib_ref is not None}
    table_df = pd.DataFrame(table_json)
    ours_row = None
    for i, cell_val in enumerate(table_df[table_df.columns[0]]):
        # extract the citation
        matches = re.search("{{cite:([a-f\d]{7})}}", cell_val)
        if matches is None:
            # we could be in an "ours" row
            if "standard" in cell_val.lower():
                # this is
                continue
            else:
                # track the last unmatched row as "ours"
                ours_row = {
                    "bib_hash_or_arxiv_id": paper_id,
                    "row": i,
                    "corpus_id": -1,  # TODO: After running `populate_bib_entries`, this should be replaced with the correct corpus id
                    # bib_entries[table_original["paper_id"]]["corpus_id"],
                    "type": "ours",
                }
        else:
            cite_id = matches[1]
            bib_hash_match = cite_id_map[cite_id]
            table_row_bib_map.append(
                {
                    "bib_hash_or_arxiv_id": bib_hash_match,
                    "row": i,
                    "corpus_id": -1,  # bib_entries[bib_hash_match]["corpus_id"],  # this will get overwritten
                    "type": "ref",
                }
            )

    if ours_row is not None:
        table_row_bib_map.append(ours_row)
    return table_row_bib_map


def create_dataset(labeled_tables, filters):
    """
    Flattens the tables_by_paper into a list of tables with associated information to create a dataset."""
    dataset = []
    if labeled_tables and labeled_tables[0].get("labels") is None:
        print("Unable to find labels on the tables. All are being converted to json.")
    missing_bib_hashes = []
    skipped_table_hashes = []
    for table_i, table in tqdm(enumerate(labeled_tables), total=len(labeled_tables)):
        should_skip_table = False

        # there should usually be labels, but there might not be, in which case don't filter anything...
        if table.get("labels") is not None:
            for filter_fn in filters:
                if not table["labels"][filter_fn.__name__]:
                    should_skip_table = True
                    break
        if should_skip_table:
            continue

        if "table_html" not in table:
            table_soup = soupify(table["xml"])
        else:
            table_soup = soupify(table["table_html"])
        cites = table_soup.find_all("cit")
        cite_shas = [cite.get("sha") for cite in cites]

        try:
            with warnings.catch_warnings(action="ignore"):
                table_json = soup_to_json(table_soup)
        except Exception as e:
            table_json = {"table_dict": {}}
            print(f"Error {e}")

        if not table_json["table_dict"]:
            print(f"Skipping {table['_table_hash']} because `soup_to_json` failed")
            skipped_table_hashes.append(table["_table_hash"])
            continue
        # trp = table_requires_paper(table_json)
        row_bib_map = get_table_row_bib_map(table_json["table_dict"], cite_shas, table["paper_id"])

        new_sample = {
            "paper_id": table["paper_id"],
            "_pdf_hash": table.get("_pdf_hash"),
            "_source_hash": table.get("_source_hash"),
            "_source_name": table.get("_source_name"),
            "_table_hash": table["_table_hash"],
        }

        if "caption" in table:
            new_sample["caption"] = table["caption"]

        if "in_text_ref" in table:
            new_sample["in_text_ref"] = table["in_text_ref"]

        new_sample |= {
            "table_html": str(table_soup),
            "table_json": table_json,
            "row_bib_map": row_bib_map,
            "bib_hash": cite_shas,
        }

        if "labels" in table:
            new_sample["labels"] = table["labels"]

        if "input_papers" in table:
            for row in row_bib_map:
                if not row["bib_hash_or_arxiv_id"] in table["input_papers"]:
                    missing_bib_hashes.append(row["bib_hash_or_arxiv_id"])
                    continue

                paper_info = table["input_papers"][row["bib_hash_or_arxiv_id"]]
                row["corpus_id"] = paper_info["corpus_id"]
                row["title"] = paper_info["title"]
                row["abstract"] = paper_info["abstract"]
        dataset.append(new_sample)

    print(f"Skipped {len(skipped_table_hashes)} tables")
    print(f"E.g. {skipped_table_hashes[:10]} ...")
    print()
    print(f"Missing {len(missing_bib_hashes)} bib_hashes or arxiv_ids")
    print(f"E.g. {missing_bib_hashes[:10]} ...")
    return dataset
    
    # for paper_i, paper in enumerate(tables_by_paper):
    #     for table_key in paper["tables"]:
    #         if "soup" in paper["tables"][table_key]:
    #             table_soup = paper["tables"][table_key]["soup"]
    #         else:
    #             table_soup = BeautifulSoup(paper["tables"][table_key]["table_html"])

    #         cites = table_soup.find_all("cit")
    #         cite_shas = [cite.get("sha") for cite in cites]

    #         table_json = soup_to_json(table_soup)
    #         if not table_json["table_dict"]:
    #             print(f"Skipping {table_key} because `soup_to_json` failed")
    #             continue
    #         # trp = table_requires_paper(table_json)
    #         row_bib_map = get_table_row_bib_map(table_json["table_dict"], cite_shas, paper["paper_id"])
    #         print(len(dataset), paper_i, table_key)
    #         dataset.append(
    #             {
    #                 "paper_id": paper["paper_id"],
    #                 "_pdf_hash": paper["_pdf_hash"],
    #                 "_source_hash": paper["_source_hash"],
    #                 "_source_name": paper["_source_name"],
    #                 "_table_hash": table_key,
    #                 "table_html": str(table_soup),
    #                 "table_json": table_json,  # this is kinda hard
    #                 # "table_requires_paper": trp,  # whether the the paper containing the table is one of the rows
    #                 "row_bib_map": row_bib_map,
    #                 "bib_hash": cite_shas,
    #             }
    #         )
    # return dataset


DEFAULT_POST_JSON_FILTERS = ["no_formula", "has_caption", "no_no_cite", "no_dup"]


def get_high_quality_tables(valid_tables_with_jsons, filters=None):
    # reloading data
    # print("reloading data...")
    # with open("arxiv_dump/out_xml_fulltext_filtered/valid_tables_json.jsonl") as f:
    #     valid_tables_with_jsons = [json.loads(line) for line in f]
    # print("Done")

    seen_table_strs = set()
    high_quality_tables = []
    if filters is None:
        filters = DEFAULT_POST_JSON_FILTERS

    valid_corpus_ids = []
    if "rows_no_missing_full_texts" in filters:
        for filename in [
            "data/v2/metric_validation_0/full_texts_corpus_ids.jsonl",
            "data/v2/metric_validation_1/full_texts_corpus_ids.jsonl",
            "data/v2/highest_quality_tables_1k/full_texts_corpus_ids.jsonl",
        ]:
            with open(filename) as f:
                valid_corpus_ids.extend([json.loads(line)["corpusId"] for line in f])

    for table_i, table in enumerate(valid_tables_with_jsons):

        table_str = "\n".join(
            [
                "\t".join([colname] + table["table_json"]["table_dict"][colname])
                for colname in table["table_json"]["table_dict"]
            ]
        )

        # remove tables with formulas
        if "no_formula" in filters and "{{formula:" in table_str:
            continue

        # ensure the table has a caption
        if "has_caption" in filters and table["caption"] == "NO_CAPTION" or not table["caption"].strip():
            continue

        # ensure table has an in-text reference
        if "has_in_text_ref" in filters and not table["in_text_ref"]:
            continue

        # for now, ignore tables that are missing citations
        if "no_no_cite" in filters and "no_cite" in table_str:
            continue

        if "max_one_no_cite" in filters and table_str.count("no_cite") > 1:
            continue

        # ignore tables that have merged headers
        if "no_merged_headers" in filters and any(
            ["-" in header for header in table["table_json"]["table_dict"].keys()]
        ):
            continue

        # if the reference column contains no citations, then skip the table as well.
        table_dict = table["table_json"]["table_dict"]
        if all([cell.strip() == "-" for cell in table_dict["References"]]):
            continue

        # if the table cites papers we don't have title & abstract for, then skip
        try:
            if "no_missing_titles" in filters and any([row["title"] is None for row in table["row_bib_map"]]):
                continue
        except KeyError:
            print("key error:", table["_table_hash"])
            continue

        if "no_missing_abstracts" in filters and any([row["abstract"] is None for row in table["row_bib_map"]]):
            continue

        # edit the rows and bibmap to remove rows missing titles and/or abstracts
        remove_rows = []
        remove_unique_hashes = set()
        new_row_bib_map = []
        for row in table["row_bib_map"]:
            should_remove_row = False
            if "rows_no_missing_titles" in filters and row["title"] is None:
                should_remove_row = True
                remove_unique_hashes.add(row["bib_hash_or_arxiv_id"])
            if "rows_no_missing_abstracts" in filters and row["abstract"] is None:
                should_remove_row = True
                remove_unique_hashes.add(row["bib_hash_or_arxiv_id"])
            if "rows_no_missing_full_texts" in filters and row["corpus_id"] not in valid_corpus_ids:
                should_remove_row = True
                remove_unique_hashes.add(row["bib_hash_or_arxiv_id"])

            if should_remove_row:
                remove_rows.append(row["row"])
            else:
                new_row = {k: v for k, v in row.items()}
                new_row["row"] = len(new_row_bib_map)
                new_row_bib_map.append(new_row)

        # ignore tables that are too small
        if (
            "more_than_two_rows" in filters
            and len(table["table_json"]["table_dict"]["References"]) - len(remove_rows) < 2
        ):
            continue

        if (
            "more_than_two_uniq_rows" in filters
            and len(set(table["table_json"]["table_dict"]["References"])) - len(remove_unique_hashes) < 2
        ):
            continue
        # do some post processing by removing columns that only contain additional citations
        remove_cols = []

        for col in table_dict.keys():
            # always remove cells that only include additional citations
            if all(
                [
                    cell.strip() == "-" or re.sub(r"(and|[,])", "", cell).strip().startswith("{{cite:")
                    for cell in table_dict[col]
                ]
            ):
                remove_cols.append(col)

            # also remove colums whose headers are just numbers...
            if re.match(r"\d+$", col) is not None:
                remove_cols.append(col)

            if "cols_no_formula" in filters and any(["{{formula:" in cell for cell in table_dict[col] + [col]]):
                remove_cols.append(col)

            if "cols_no_formula_colname" in filters and "{{formula:" in col:
                remove_cols.append(col)

            if "cols_no_names" in filters and col.lower() in {
                "reference",
                "author",
                "reference-(5)",
                "author/reference",
            }:
                remove_cols.append(col)
            if (
                "col_no_generic" in filters
                and col
                in {
                    "Venue",
                    "Month",
                    "Year",
                    "No.",
                    "[HTML]D0CECE\nYear",
                    "Title",
                    "URL",
                    "[HTML]BBDAFFYear",
                    "Link",
                    "Version",
                    "Organiser",
                    "Citations",
                    "Pub.",
                    "Title of Survey Article",
                    "License",
                    "Publication",
                }
                or col.lower
                in {
                    "reference",
                    "author",
                    "reference-(5)",
                    "author/reference",
                }
            ):
                remove_cols.append(col)

            if "cols_no_old_citation_col" in filters:
                remove_cols.append(table["table_json"]["old_citation_column"])

            if "cols_no_float" in filters and any([FLOAT_REGEX.search(cell) for cell in table_dict[col] + [col]]):
                remove_cols.append(col)

            if "cols_no_figure" in filters and any([r"{{figure" in cell for cell in table_dict[col] + [col]]):
                remove_cols.append(col)

            if "cols_no_ent_or_gen" in filters or "cols_no_numeric" in filters:
                aspect_type = get_aspect_type(table_dict[col])
                if "cols_no_numeric" in filters and aspect_type == "num":
                    remove_cols.append(col)
                if "cols_no_ent_or_gen" in filters and aspect_type in {"gen", "ent"}:
                    remove_cols.append(col)

            if "cols_no_numeric" in filters:
                pass

        new_table_dict = {}
        for col in table_dict.keys():
            if col in remove_cols and col != "References":
                continue

            new_table_dict[col] = [val for row_i, val in enumerate(table_dict[col]) if row_i not in remove_rows]

        # filter out tables that don't have enough columns (references, something, something) - need at least three
        min_rows = 2 if "cols_no_ent_or_gen" in filters or "cols_no_numeric" in filters else 3
        if len(new_table_dict) < min_rows:
            continue

        # don't add tables that have already been added
        if "no_dup" in filters:
            table_str_no_refs = "\n".join(
                [
                    re.sub(r"{{cite:.{7}}}", "[cite]", "\t".join([colname] + new_table_dict[colname]))
                    for colname in new_table_dict
                    if colname != "References"
                ]
            )
            if table_str_no_refs in seen_table_strs:
                continue
            else:
                seen_table_strs.add(table_str_no_refs)

        table_cp = copy.deepcopy(table)
        table_cp["table_json"]["table_dict"] = new_table_dict
        table_cp["row_bib_map"] = new_row_bib_map
        high_quality_tables.append(table_cp)

    return high_quality_tables


def run(
    in_path,
    out_labeled_path,
    out_filtered_path,
    out_high_quality_path,
    out_high_quality_schemes_path,
    out_mid_quality_path,
    should_label,
    should_filter,
    should_create_quality_datasets,
):
    # labeling
    if should_label:
        assert out_labeled_path is not None
        labeled_tables = extract_valid_tables(in_path, DEFAULT_TABLE_LABELS, label_tables=True)
        labeled_tables_dataset = []
        for paper_i, paper in enumerate(labeled_tables):

            # get in-text references
            in_text_refs_by_table_id = defaultdict(list)
            for section in paper["body_text"]:
                if section["ref_spans"] and section["content_type"] == "paragraph":
                    for ref in section["ref_spans"]:
                        if ref["ref_id"] in paper["tables"]:
                            in_text_refs_by_table_id[ref["ref_id"]].append(section)

            for table_key in paper["tables"]:
                labeled_tables_dataset.append(
                    {
                        "paper_id": paper["paper_id"],
                        "_pdf_hash": paper["_pdf_hash"],
                        "_source_hash": paper["_source_hash"],
                        "_source_name": paper["_source_name"],
                        "_table_hash": table_key,
                        "table_html": str(paper["tables"][table_key]["soup"]),
                        "labels": paper["tables"][table_key]["labels"],
                        "caption": paper["tables"][table_key]["caption"],
                        "in_text_ref": in_text_refs_by_table_id[table_key],
                    }
                )
        with open(out_labeled_path, "w") as f:
            for sample in labeled_tables_dataset:
                f.write(json.dumps(sample) + "\n")

    elif should_filter:
        # assumes that `in_path` has labels
        if os.path.splitext(in_path)[1] == ".gz":
            f = gzip.open(in_path, "r")
        else:
            f = open(in_path)
        # with open(in_path) as f:
        labeled_tables_dataset = [json.loads(line) for line in f]
        f.close()

    # filtering
    if should_filter:
        assert out_filtered_path is not None
        filtered_tables_dataset = create_dataset(labeled_tables_dataset, DEFAULT_TABLE_FILTERS)

        with open(out_filtered_path, "w") as f:
            for sample in filtered_tables_dataset:
                f.write(json.dumps(sample) + "\n")
    elif should_create_quality_datasets:
        # assumes that `in_path` has jsons
        with open(in_path) as f:
            filtered_tables_dataset = [json.loads(line) for line in f]

    if should_create_quality_datasets:
        if out_high_quality_path is not None:
            filters_hq = [
                "no_formula",
                "has_caption",
                "no_no_cite",
                "no_dup",
                "more_than_two_uniq_rows",
                "has_in_text_ref",
                "cols_no_names",
                "cols_no_old_citation_col",
                "no_merged_headers",
                "cols_no_float",
                "cols_no_figure",
                "rows_no_missing_titles",
                "rows_no_missing_abstracts",
                "rows_no_missing_full_texts",
            ]

            # ["no_formula", "has_caption", "no_no_cite", "no_dup", "more_than_two_uniq_rows", "has_in_text_ref", "cols_no_names", "cols_no_old_citation_col", "no_merged_headers", "cols_no_float", "cols_no_figure", "rows_no_missing_titles", "rows_no_missing_abstracts"]
            # ["no_formula", "has_caption", "no_no_cite", "no_dup", "more_than_two_uniq_rows", "has_in_text_ref", "cols_no_names", "cols_no_old_citation_col", "no_merged_headers", "cols_no_float", "cols_no_figure", "rows_no_missing_titles", "rows_no_missing_abstracts"]
            # filters_hq = [
            #     "no_formula",
            #     "has_caption",
            #     "no_no_cite",
            #     "no_dup",
            #     "more_than_two_uniq_rows",
            #     "has_in_text_ref",
            #     "cols_no_names",
            #     "cols_no_old_citation_col",
            #     "no_merged_headers",
            #     "cols_no_float",
            #     "cols_no_figure",
            #     "no_missing_titles",
            #     "no_missing_abstracts",
            # ]
            # filters_hq = [
            #     "no_formula",
            #     "has_caption",
            #     "no_no_cite",
            #     "no_dup",
            #     "more_than_two_uniq_rows",
            #     "has_in_text_ref",
            #     "cols_no_names",
            #     "cols_no_old_citation_col",
            #     "no_merged_headers",
            #     "cols_no_float",
            #     "cols_no_figure",
            #     "no_missing_titles",
            #     "no_missing_abstracts",
            #     "cols_no_ent_or_gen",
            #     # "cols_no_numeric",
            # ]
            # filters_hq = [
            #     "no_formula",
            #     "has_caption",
            #     "no_no_cite",
            #     "no_dup",
            #     # "more_than_two_rows",
            #     "more_than_two_uniq_rows",
            #     "has_in_text_ref",
            #     "cols_no_names",
            #     "cols_no_old_citation_col",
            #     "no_merged_headers",
            #     # asdf
            #     "cols_no_float",
            #     "cols_no_figure",
            #     "rows_no_missing_titles",
            #     "rows_no_missing_abstracts",
            #     # "no_missing_titles",
            #     # "no_missing_abstracts",
            # ]
            dataset_hq = get_high_quality_tables(filtered_tables_dataset, filters_hq)
            print("Size of High Quality dataset:", len(dataset_hq))
            with open(out_high_quality_path, "w") as f:
                for sample in dataset_hq:
                    f.write(json.dumps(sample) + "\n")

            with open(os.path.splitext(out_high_quality_path)[0] + "_filters.json", "w") as f:
                json.dump(filters_hq, f)

        if out_high_quality_schemes_path is not None:
            filters_high_quality_schemes = [
                "no_dup",
                # "max_one_no_cite",
                # "more_than_two_rows",
                "more_than_two_uniq_rows",
                "has_caption",
                "cols_no_names",
                # "cols_no_formula",
                "cols_no_formula_colname",
                # "cols_no_old_citation_col",
                "no_merged_headers",
                # "cols_no_float",
                # "cols_no_figure",
                # "rows_no_missing_titles",
                # "rows_no_missing_abstracts",
            ]
            dataset_high_quality_schemes = get_high_quality_tables(
                filtered_tables_dataset,
                filters_high_quality_schemes,
            )
            print("Size of dataset_high_quality_schemes:", len(dataset_high_quality_schemes))
            with open(out_high_quality_schemes_path, "w") as f:
                for sample in dataset_high_quality_schemes:
                    f.write(json.dumps(sample) + "\n")

            with open(os.path.splitext(out_high_quality_schemes_path)[0] + "_filters.json", "w") as f:
                json.dump(filters_high_quality_schemes, f)

        if out_mid_quality_path is not None:
            filters_mid_quality = [
                "no_dup",
                "max_one_no_cite",
                # "more_than_two_rows",
                "more_than_two_uniq_rows",
                "has_caption",
                "cols_no_names",
                "cols_no_formula",
                "cols_no_float",
                "cols_no_figure",
                "rows_no_missing_titles",
                "rows_no_missing_abstracts",
                # "cols_no_old_citation_col",
            ]
            dataset_mid_quality_tables = get_high_quality_tables(filtered_tables_dataset, filters_mid_quality)
            print("Size of dataset_mid_quality_tables:", len(dataset_mid_quality_tables))
            print(f"Saved to {out_mid_quality_path}")
            with open(out_mid_quality_path, "w") as f:
                for sample in dataset_mid_quality_tables:
                    f.write(json.dumps(sample) + "\n")

            with open(os.path.splitext(out_mid_quality_path)[0] + "_filters.json", "w") as f:
                json.dump(filters_mid_quality, f)


BLACK_LIST = [
    "done.log",
    "log.txt",
    # "2211.jsonl.gz",
    # "2212.jsonl.gz",
    # "2306.00000-17847v1.jsonl.gz",
    # "2308.00000-16912v1.jsonl.gz",
    # "2307.jsonl.gz",
]


def main():
    argp = ArgumentParser()
    argp.add_argument("in_path", type=str)
    argp.add_argument("--out_labeled_path", type=str)
    argp.add_argument("--out_filtered_path", type=str)
    argp.add_argument("--out_high_quality_path", type=str)
    argp.add_argument("--out_high_quality_schemes_path", type=str)
    argp.add_argument("--out_mid_quality_path", type=str)
    argp.add_argument("--label", action="store_true")
    argp.add_argument("--filter", action="store_true")
    argp.add_argument("--create_quality_datasets", action="store_true")
    argp.add_argument("--num_processes", type=int, default=1)
    args = argp.parse_args()

    if not args.label and not args.filter and not args.create_quality_datasets:
        args.label = True
        args.filter = True
        args.create_quality_datasets = True

    if args.num_processes == 1:
        run(
            args.in_path,
            args.out_labeled_path,
            args.out_filtered_path,
            args.out_high_quality_path,
            args.out_high_quality_schemes_path,
            args.out_mid_quality_path,
            args.label,
            args.filter,
            args.create_quality_datasets,
        )
    else:
        # assume in_path contains the in_paths we care about
        with multiprocessing.Pool(processes=args.num_processes) as pool:
            results = []
            for in_path in os.listdir(args.in_path):
                if in_path in BLACK_LIST:
                    print(f"Skipping {in_path}")
                    continue
                out_labeled_path = (
                    os.path.join(args.out_labeled_path, in_path.split(".")[0]) + ".jsonl"
                    if args.out_labeled_path is not None
                    else None
                )
                out_filtered_path = (
                    os.path.join(args.out_filtered_path, in_path.split(".")[0]) + ".jsonl"
                    if args.out_filtered_path is not None
                    else None
                )
                out_high_quality_path = (
                    os.path.join(args.out_high_quality_path, in_path.split(".")[0]) + ".jsonl"
                    if args.out_high_quality_path is not None
                    else None
                )
                out_high_quality_schemes_path = (
                    os.path.join(args.out_high_quality_schemes_path, in_path.split(".")[0]) + ".jsonl"
                    if args.out_high_quality_schemes_path is not None
                    else None
                )
                out_mid_quality_path = (
                    os.path.join(args.out_mid_quality_path, in_path.split(".")[0]) + ".jsonl"
                    if args.out_mid_quality_path is not None
                    else None
                )
                print("In:", os.path.join(args.in_path, in_path))
                print("out:", out_labeled_path)
                runner = functools.partial(
                    run,
                    os.path.join(args.in_path, in_path),
                    out_labeled_path,
                    out_filtered_path,
                    out_high_quality_path,
                    out_high_quality_schemes_path,
                    out_mid_quality_path,
                    args.label,
                    args.filter,
                    args.create_quality_datasets,
                )
                result = pool.apply_async(runner)
                results.append(result)
            for result in results:
                result.get()
            pool.close()
            pool.join()


if __name__ == "__main__":
    main()

