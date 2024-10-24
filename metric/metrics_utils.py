"""Utility functions for computing metrics"""

import difflib
import json
import os
import re
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer

from table import Table

stopwords = stopwords.words("english")
punctuation = "()[]{},.?!/''\"``"
ps = PorterStemmer()

# Moving to Together AI API to query mistral for decontextualization
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)


class BaseFeaturizer:
    """Given a list of columns, create featurized strings for every column, for better matching/alignment.

    Attributes:
        name (str): The name of the featurization strategy to be used.
        metadata (dict): Dictionary containing any hyperparameter settings required.
    """

    name: str
    metadata: dict

    def __init__(self, name):
        """Initialize the featurizer.

        By default, metadata is an empty dictionary
        """

        self.name = name
        self.metadata = {}

    def featurize(self, column_names: list[str], table: Table) -> list[str]:
        """Given a list of columns, return a list of featurized strings (one per column).
        Base featurizer simply returns the column names as-is.
        Other featurizers should re-implement this method.

         Args:
            column_names (list[str]): List of column names to featurize
            table (Table): Table containing provided column names
        """
        return column_names


class ValueFeaturizer(BaseFeaturizer):
    """Value featurizer featurizes columns by adding values in addition to column name.
    No additional metadata.
    """

    name: str
    metadata: dict

    def __init__(self, name):
        super().__init__(name)

    def featurize(self, column_names: list[str], table: Table) -> list[str]:
        """Return featurized strings containing column values"""
        featurized_columns = []
        for column in column_names:
            value_list = []
            for value in list(table.values[column].values()):
                if isinstance(value, list):
                    value_list += [str(x) for x in value]
                else:
                    value_list.append(str(value))
            column_values = ", ".join(value_list)
            featurized_columns.append(f"Column named {column} has values: {column_values}")
        return featurized_columns


class DecontextFeaturizer(BaseFeaturizer):
    """Decontextualization featurizer featurizes columns by using column names and values
    to generate a more detailed description of the type of information being captured.
    Metadata includes the tokenizer and model to be used to produce these descriptions.
    """

    name: str
    metadata: dict

    def __init__(self, name, model="mistralai/Mistral-7B-Instruct-v0.2"):
        super().__init__(name)
        self.metadata["model_name"] = model
        self.load_model_and_tokenizer(model)

    def load_model_and_tokenizer(self, model_name: str):
        """Given a model name, start a together client to query that model.

        Args:
           model_name (str): Name of model to query
        """
        self.metadata["model"] = OpenAI(
            api_key=TOGETHER_API_KEY,
            base_url="https://api.together.xyz/v1",
        )
        # mistral_tokenizer = AutoTokenizer.from_pretrained(model_name)
        # mistral_tokenizer.pad_token = mistral_tokenizer.eos_token
        # mistral_model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     load_in_8bit=True,
        # )
        # mistral_model.config.pad_token_id = mistral_model.config.eos_token_id
        # self.metadata["model"] = mistral_model
        # self.metadata["tokenizer"] = mistral_tokenizer

    def query_model(self, prompt):
        """Run model inference on provided prompt.

        Args:
            prompt (str): Prompt to query model with.
        """
        # generated_ids = []
        # inputs = self.metadata['tokenizer'].apply_chat_template(prompt, return_tensors="pt").to(DEVICE)
        try:
            chat_completion = self.metadata["model"].chat.completions.create(
                messages=prompt,
                model=self.metadata["model_name"],
                max_tokens=256,
                temperature=0.7,
                top_p=0.7,
            )
            response = chat_completion.choices[0].message.content
            # generated_ids = self.metadata['model'].generate(inputs, max_new_tokens=100, do_sample=True, num_return_sequences=1)
        except Exception as e:
            print(e)
            time.sleep(10)
            return self.query_model(prompt)
        #     response = self.metadata['tokenizer'].batch_decode(generated_ids[:, inputs.shape[1] :], skip_special_tokens=True)
        # except torch.cuda.OutOfMemoryError:
        #     # for debugging
        #     print("oom:", inputs.shape, flush=True)
        #     raise torch.cuda.OutOfMemoryError
        # finally:
        #     # to avoid taking up gpu memory
        #     del inputs
        #     del generated_ids
        #     torch.cuda.empty_cache()

        return response

    def create_column_decontext_prompts(self, column_names: list[str], table: pd.DataFrame) -> list[str]:
        """Construct a list of prompts to decontextualize all column names present in the table.

        Args:
            column_names (list[str]): List of column names to construct decontextualization prompts for.
            table (pd.DataFrame): Source table to provide additional context (in dataframe format).
        """
        decontext_prompts = []
        for column in column_names:
            # cur_table = table[[column]]
            instruction = f"""\
                In the context of the following table from a scientific paper, what does {column} refer to? Answer in a single sentence. If the answer is not clear just write 'unanswerable'.
                Table:
                {table.to_markdown()}\
            """
            decontext_prompts.append(instruction)
        return decontext_prompts

    # TODO: Can we skip filtering out of numeric/binary values now that we aren't decontextualizing values?
    # TODO: Based on prior discussions, I'm not using paper title/abstract/section text/caption during decontextualization,
    # since we may not accurately get this information for predicted tables. We can revisit this after seeing what scores look like?

    def featurize(self, column_names: list[str], table: Table) -> list[str]:
        """Return featurized strings containing column values"""
        # If decontextualization has already been computed and stored,
        # return the cached descriptions instead of regenerating
        if table.decontext_schema is not None:
            return [table.decontext_schema[x] for x in column_names]
        featurized_columns = []
        table_df = pd.DataFrame(table.values)
        column_decontext_prompts = self.create_column_decontext_prompts(column_names, table_df)
        for i, prompt in enumerate(column_decontext_prompts):
            full_prompt = [{"role": "user", "content": prompt}]
            try:
                response = self.query_model(full_prompt)
            except torch.cuda.OutOfMemoryError:
                # If prompt doesn't fit in memory, just return column name
                print("OOM num chars:", len(prompt))
                response = [column_names[i]]
            featurized_columns.append(response.strip())
            # featurized_columns.append(response[0])
        return featurized_columns


class BaseAlignmentScorer:
    """Computes and returns an alignment score matrix between all column pairs given a pair of tables.

    Attributes:
        name (str): The name of the method to be used for alignment.
        metadata (dict): Dictionary containing any hyperparameter/threshold values required for the alignment method.
    """

    name: str
    metadata: dict

    def __init__(self, name):
        """Initialize the alignment method.

        By default, metadata is an empty dictionary
        """

        self.name = name
        self.metadata = {}

    # This function must be implemented for each alignment method sub-class
    def calculate_pair_similarity(self, prediction: str, target: str):
        """Calculate the score for the the given (prediction, target) string pair.

        Args:
            prediction (str): The string generated by the model.
            target (str): The gold string.
        """

        raise NotImplementedError()

    # Function to compute alignment scores for all column pairs, given a pair of tables
    def score_schema_alignments(
        self, pred_table: Table, gold_table: Table, featurizer=BaseFeaturizer("name")
    ) -> dict[tuple, float]:
        """Given a pair of tables, calculate similarity scores for all possible schema alignments (i.e., all pairs of columns)

        Args:
           pred_table (Table): The table generated by the model.
           gold_table (Table): The gold table.
           featurizer (Featurizer): Featurization strategy to be applied to columns (default simply uses column names)
        """
        alignment_matrix = {}
        pred_col_list = list(pred_table.schema)
        gold_col_list = list(gold_table.schema)

        # Apply specified featurization strategy before computing alignment
        featurized_pred_col_list = featurizer.featurize(pred_col_list, pred_table)
        featurized_gold_col_list = featurizer.featurize(gold_col_list, gold_table)

        # For certain alignment methods that use neural models (like sentence transformer),
        # to improve efficiency, calculate_pair_similarity operates in batch mode (on lists of strings).
        # So alignment matrix construction differs slightly for both categories.
        if self.name not in ["sentence_transformer"]:
            for i, gold_col_name in enumerate(featurized_gold_col_list):
                for j, pred_col_name in enumerate(featurized_pred_col_list):
                    pair_score = self.calculate_pair_similarity(pred_col_name, gold_col_name)
                    alignment_matrix[(gold_col_list[i], pred_col_list[j])] = pair_score
        else:
            # Instead of computing similarity for every column pair separately, the computation is batched.
            # This ensures that encoding is performed only once instead of being recomputed per comparison.
            sim_matrix = self.calculate_pair_similarity(featurized_pred_col_list, featurized_gold_col_list)
            for i, gold_col_name in enumerate(featurized_gold_col_list):
                for j, pred_col_name in enumerate(featurized_pred_col_list):
                    alignment_matrix[(gold_col_list[i], pred_col_list[j])] = sim_matrix[j][i]

        return alignment_matrix
    
    # Function to compute alignment scores for all column pairs, given a pair of tables
    def score_value_alignments(
        self, pred_table: Table, gold_table: Table
    ) -> dict[tuple, float]:
        """Given a pair of tables with matched columns, calculate similarity scores for all pairs of values under those columns.
           Currently, this does not apply any featurization to individual values.

        Args:
           pred_table (Table): The table generated by the model.
           gold_table (Table): The gold table.
        """
        alignment_matrix = {}
        gold_col_list = list(gold_table.schema)

        # Iterate over the matched column names 
        # For each column name, compute alignment scores per pair of rows
        pair_index = 0
        for i, gold_col_name in enumerate(gold_col_list):
            for j, corpus_id in enumerate(gold_table.values[gold_col_name]):
                pair_index += 1
                gold_value = gold_table.values[gold_col_name][corpus_id][0]
                if not gold_col_name in pred_table.values or not corpus_id in pred_table.values[gold_col_name]:
                    alignment_matrix[(gold_col_name, corpus_id)] = 0.0
                    continue
                if pred_table.values[gold_col_name][corpus_id] == "N/A":
                    continue
                pred_value = pred_table.values[gold_col_name][corpus_id]
                pair_score = self.calculate_pair_similarity(pred_value, gold_value)
                # Set a counter/ID as the first element of the key in alignment matrix
                # since this element will be used as a unique ID when aggregating for recall
                alignment_matrix[(pair_index, gold_col_name, corpus_id)] = pair_score

        return alignment_matrix


class ExactMatchScorer(BaseAlignmentScorer):
    """Exact match scorer has no additional metadata."""

    def __init__(self):
        super().__init__("exact_match")

    def calculate_pair_similarity(self, prediction: str, target: str) -> float:
        """Similarity calculation based on exact string match."""
        if prediction.lower() == target.lower():
            return 1.0
        return 0.0


class EditDistanceScorer(BaseAlignmentScorer):
    """Edit distance scorer has no additional metadata."""

    def __init__(self):
        super().__init__("edit_distance")

    def calculate_pair_similarity(self, prediction: str, target: str) -> float:
        """Similarity calculation based on edit distance.
        We compute the edit distance between two strings using difflib.
        """
        matcher = difflib.SequenceMatcher(None, prediction.lower(), target.lower())
        return float(matcher.ratio())


class JaccardAlignmentScorer(BaseAlignmentScorer):
    """Alignment scorer which uses Jaccard similarity to compare schemas.
    Metadata includes a flag which can determine whether to use stopwords
    during Jaccard similarity computation.

    """

    def __init__(self, remove_stopwords=True):
        """We can choose whether to use or ignore stopwords while computing Jaccard similarity."""
        super().__init__("jaccard")
        self.metadata["remove_stopwords"] = remove_stopwords

    def get_keywords(self, sentence: str) -> set[str]:
        """Extract non-stopword keywords from a sentence.

        Extract keywords from a sentence by lowercasing, tokenizing and stemming words. Punctuation and
        stopwords ar filtered out. Only unique words are returned.

        Args:
            sentence (str): The text to extract keywords from.

        Returns:
            set[str] containing the keywords in the sentence.
        """
        return set(
            [
                ps.stem(word.lower())
                for word in word_tokenize(sentence)
                if word not in stopwords and word not in punctuation
            ]
        )

    def jaccard(self, a: set[Any], b: set[Any]) -> float:
        """Calculate and return the Jaccard similarity between set `a` and `b`."""

        return len(a & b) / len(a | b)

    def calculate_pair_similarity(self, prediction: str, target: str) -> float:
        """Similarity calculation based on jaccard overlap between tokens."""
        prediction_words, target_words = [], []
        if self.metadata["remove_stopwords"]:
            prediction_words = self.get_keywords(prediction)
            target_words = self.get_keywords(target)
        else:
            prediction_words = set(word_tokenize(prediction))
            target_words = set(word_tokenize(target))
        return self.jaccard(prediction_words, target_words)


class SentenceTransformerAlignmentScorer(BaseAlignmentScorer):
    """Alignment scorer which uses similarity from sentence transformer embeddings to compare schemas.
    Metadata contains a variable specifying which pretrained model to use
    See: https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models
    Fast but worse: all-MiniLM-L6-v2
    Slow but better: all-mpnet-base-v2
    These are not specifically trained on scientific text.

    """

    def __init__(self, model="all-MiniLM-L6-v2"):
        """We can choose which sentence transformer model to use while initializing."""
        super().__init__("sentence_transformer")
        self.metadata["model"] = model
        self.model = SentenceTransformer(model)

    # For better efficiency, the pair similarity calculation function takes batches of strings
    def calculate_pair_similarity(self, predictions: list[str], targets: list[str]) -> float:
        """Similarity calculation based on jaccard overlap between tokens."""
        pred_embeds = self.model.encode(predictions)
        gold_embeds = self.model.encode(targets)
        sim_mat_cosine = util.cos_sim(pred_embeds, gold_embeds).numpy()
        return sim_mat_cosine


# The functions below have been useful in past projects and might also be useful in this one,
# so I've included them below.


def get_p_r_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Get the precision, recall, and F1 given true positives, false positives, and false negatives.

    Args:
        tp (int): True positives.
        fp (int): False positives.
        fn (int): False negatives.

    Returns:
        Tuple[float, float, float] containing (precision, recall, F1).
    """

    if tp == 0:
        return 0, 0, 0
    else:
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        return p, r, 2 * p * r / (p + r)


class Llama3AlignmentScorer(BaseAlignmentScorer):

    def __init__(self, name="llama", debug=False):
        super().__init__(name)

        import together
        from together import Together, error

        from .llama_aligner import PROMPT

        self.prompt_prefix = PROMPT
        self.client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        self.api_error = error.APIError
        self.debug = debug
        self._together = together

    def query_llama(self, prompt, max_tokens=200):
        response = self.client.chat.completions.create(
            model="meta-llama/Llama-3-70b-chat-hf",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers in JSON.",
                },
                {"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return response

    def score_schema_alignments(
        self, pred_table: Table, gold_table: Table, featurizer=BaseFeaturizer("name")
    ) -> dict[tuple, float]:
        """Given a pair of tables, calculate similarity scores for all possible schema alignments (i.e., all pairs of columns)

        Args:
           pred_table (Table): The table generated by the model.
           gold_table (Table): The gold table.
           featurizer (Featurizer): Featurization strategy to be applied to columns (default simply uses column names)
        """
        alignment_matrix = {}
        pred_col_list = list(pred_table.values.keys())
        gold_col_list = list(gold_table.values.keys())

        # Apply specified featurization strategy before computing alignment
        featurized_pred_col_list = featurizer.featurize(pred_col_list, pred_table)
        featurized_gold_col_list = featurizer.featurize(gold_col_list, gold_table)

        # replace the column headers with the featurized ones
        new_pred_table = {
            new_key: value for new_key, value in zip(featurized_pred_col_list, pred_table.values.values())
        }
        new_gold_table = {
            new_key: value for new_key, value in zip(featurized_gold_col_list, gold_table.values.values())
        }

        prompt = (
            self.prompt_prefix
            + f"""
Table 1:
{pd.DataFrame(new_gold_table).to_markdown()}

Table 2:
{pd.DataFrame(new_pred_table).to_markdown()}
"""
        )

        # parse out the json
        try:
            response = self.query_llama(prompt)
        except self.api_error:
            response = self.query_llama(prompt)

        alignment_str = response.choices[0].message.content
        alignment_str = alignment_str.split("Table 1:\n|")[0]
        try:
            alignment_json = json.loads(re.search("(\[.+\])", alignment_str, re.DOTALL)[0])
        except json.JSONDecodeError:
            # try again
            if response.choices[0].finish_reason == self._together.types.common.FinishReason.Length:
                response = self.query_llama(prompt, max_tokens=1000)
            else:
                response = self.query_llama(prompt)
            if self.debug:
                print(response)
            alignment_str = response.choices[0].message.content
            alignment_str = alignment_str.split("Table 1:\n|")[0]
            alignment_json = json.loads(re.search("(\[.+\])", alignment_str, re.DOTALL)[0])

        for gold_col_name in featurized_gold_col_list:
            for pred_col_name in featurized_pred_col_list:
                pair = (gold_col_name, pred_col_name)
                alignment_matrix[pair] = 1.0 if pair in alignment_json else 0.0

        alignment_matrix |= {tuple(pair): 1.0 for pair in alignment_json if pair}
        if self.debug:
            print(alignment_json)
            print(alignment_matrix)

        return alignment_matrix


# ---------- TO BE POTENTIALLY DELETED AFTER FINALIZING CODE REFACTOR -----------
# def jaccard_sentence(sentence_1: str, sentence_2: str, keywords: bool = True) -> float:
#     """Calculate the Jaccard similarity between sentences.

#     If `keywords` is True, only calculate the similarity using the keywords. If False, then use all of the tokens
#     as tokenized by nltk.word_tokenize.

#     Args:
#         sentence_1 (str): The first sentence.
#         sentence_2 (str): The second sentence.
#         keywords (bool): Whether to use keywords (True) or all tokens (False). Default is True.

#     Returns:
#         The Jaccard similarity between the two sentences (float).
#     """

#     if keywords:
#         sentence_1_tokens = get_keywords(sentence_1)
#         sentence_2_tokens = get_keywords(sentence_2)
#     else:
#         sentence_1_tokens = set(word_tokenize(sentence_1))
#         sentence_2_tokens = set(word_tokenize(sentence_2))
#     return jaccard(sentence_1_tokens, sentence_2_tokens)

# def align_schema(target, prediction, method_name, sim_threshold, **config) -> tuple[dict, np.array]:
#     """
#     Externally avaiable method used for aligning predicted and gold target tables' schema (column names)

#     config: other details about the method

#     Returns a tuple. The first element is a dict whose keys are columns in the *target* table and whose
#         values are a list of columns in the *prediction* table whose scores match. The second element is
#         a 2-d np.array with shape of (len(target), len(prediction)) that contains the pairwise scores
#         between each target and prediction column name
#     """
#     if "sentence_transformer" in method_name:

#         model = SentenceTransformer(config.get("model", "all-MiniLM-L6-v2"))

#         if method_name == "sentence_transformer-columns":
#             return _get_alignment_columns_st(model, target, prediction, sim_threshold)
#         elif method_name == "sentence_transformer-values":
#             return _get_alignment_values_st(model, target, prediction, sim_threshold)
#     elif method_name == "jaccard":
#         return _get_alignment_columns_jaccard(target, prediction, sim_threshold)

# def _get_alignment_columns_st(model, gold_table, pred_table, threshold):
#     """
#     Align tables based on the columns by constructing a score matrix
#     """
#     embeddings_gold = model.encode(list(gold_table.keys()))
#     embeddings_pred = model.encode(list(pred_table.keys()))

#     sim_mat_cosine = util.cos_sim(embeddings_gold, embeddings_pred).numpy()

#     # calculate the alignment
#     return _get_alignment(gold_table, pred_table, sim_mat_cosine, threshold)


# def _get_alignment_columns_jaccard(gold_table, pred_table, threshold):
#     """Constructs a score matrix of jaccard similarities"""
#     score_matrix = np.array(
#         [
#             [jaccard_sentence(gold_colname, pred_colname) for pred_colname in pred_table.keys()]
#             for gold_colname in gold_table.keys()
#         ]
#     )

#     return _get_alignment(gold_table, pred_table, score_matrix, threshold)

# def _get_alignment_values_st(model, gold_table, pred_table, threshold):
#     """
#     Align tables based on the values in the tables
#     """
#     embeddings_gold = model.encode(
#         [" ".join(gold_table[c][paper_id]) for c in gold_table for paper_id in gold_table[c]]
#     )
#     embeddings_pred = model.encode(
#         [" ".join(pred_table[c][paper_id]) for c in pred_table for paper_id in pred_table[c]]
#     )

#     sim_mat_cosine = util.cos_sim(embeddings_gold, embeddings_pred)

#     # next, iterate through the num_papers x num_papers blocks to aggregate similarities among papers
#     num_papers = len(list(gold_table.values())[0])
#     column_scores = np.zeros((len(gold_table), len(pred_table)))

#     for i in range(0, len(sim_mat_cosine), num_papers):
#         for j in range(0, len(sim_mat_cosine[i]), num_papers):
#             column_scores[i // num_papers][j // num_papers] = np.mean(
#                 sim_mat_cosine[i : i + num_papers, j : j + num_papers].numpy()
#             )

#     # calculate the alignment
#     return _get_alignment(gold_table, pred_table, column_scores, threshold)

# def get_similar_sentence(query: str, pool: list[str], method="sentence_transformer") -> tuple[str, float]:
#     """
#     Returns the most similar string in the pool to the query along with the similarity scores.
#     Uses
#     """
#     if method == "sentence_transformer":
#         # See: https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models
#         # Fast but worse: all-MiniLM-L6-v2
#         # Slow but better: all-mpnet-base-v2
#         # not sure how these do with scientific text...
#         model = SentenceTransformer("all-MiniLM-L6-v2")
#         embeddings = model.encode([query] + pool, convert_tensors=True)
#         scores = util.cos_sim(embeddings[0], embeddings[1:]).numpy()
#         max_sent_idx = np.argmax(scores)
#         return pool[max_sent_idx], scores[max_sent_idx]

#     elif method in ("jaccard_keywords", "jaccard"):
#         keywords = method == "jaccard_keywords"
#         scores = [jaccard_sentence(query, sentence, keywords=keywords) for sentence in pool]
#         max_sent_idx = np.argmax(scores)
#         return pool[max_sent_idx], scores[max_sent_idx]

#     else:
#         raise ValueError(f"Invalid similarity method: {method}")
