"""Utility functions for computing metrics"""
from typing import Any

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
from sentence_transformers import SentenceTransformer, util

stopwords = stopwords.words("english")
punctuation = "()[]{},.?!/''\"``"
ps = PorterStemmer()


def get_keywords(sentence: str) -> set[str]:
    """Extract keywords from a sentence.

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


def jaccard_sentence(sentence_1: str, sentence_2: str, keywords: bool = True) -> float:
    """Calculate the Jaccard similarity between sentences.

    If `keywords` is True, only calculate the similarity using the keywords. If False, then use all of the tokens
    as tokenized by nltk.word_tokenize.

    Args:
        sentence_1 (str): The first sentence.
        sentence_2 (str): The second sentence.
        keywords (bool): Whether to use keywords (True) or all tokens (False). Default is True.

    Returns:
        The Jaccard similarity between the two sentences (float).
    """

    if keywords:
        sentence_1_tokens = get_keywords(sentence_1)
        sentence_2_tokens = get_keywords(sentence_2)
    else:
        sentence_1_tokens = set(word_tokenize(sentence_1))
        sentence_2_tokens = set(word_tokenize(sentence_2))
    return jaccard(sentence_1_tokens, sentence_2_tokens)


def jaccard(a: set[Any], b: set[Any]) -> float:
    """Calculate and return the Jaccard similarity between set `a` and `b`."""

    return len(a & b) / len(a | b)


def _get_alignment(gold_table, pred_table, column_scores, threshold):
    """
    Helper function for calculating alignments.


    """
    alignment = {}
    for gold_col_i, gold_col_name in enumerate(gold_table):
        alignment[(gold_col_name, gold_col_i)] = []
        for pred_col_i, pred_col_name in enumerate(pred_table):
            if column_scores[gold_col_i, pred_col_i] > threshold:
                alignment[(gold_col_name, gold_col_i)].append((pred_col_name, pred_col_i))

    return alignment, column_scores


def _get_alignment_values_st(model, gold_table, pred_table, threshold):
    """
    Align tables based on the values in the tables
    """
    embeddings_gold = model.encode(
        [" ".join(gold_table[c][paper_id]) for c in gold_table for paper_id in gold_table[c]]
    )
    embeddings_pred = model.encode(
        [" ".join(pred_table[c][paper_id]) for c in pred_table for paper_id in pred_table[c]]
    )

    sim_mat_cosine = util.cos_sim(embeddings_gold, embeddings_pred)

    # next, iterate through the num_papers x num_papers blocks to aggregate similarities among papers
    num_papers = len(list(gold_table.values())[0])
    column_scores = np.zeros((len(gold_table), len(pred_table)))

    for i in range(0, len(sim_mat_cosine), num_papers):
        for j in range(0, len(sim_mat_cosine[i]), num_papers):
            column_scores[i // num_papers][j // num_papers] = np.mean(
                sim_mat_cosine[i : i + num_papers, j : j + num_papers].numpy()
            )

    # calculate the alignment
    return _get_alignment(gold_table, pred_table, column_scores, threshold)


def _get_alignment_columns_st(model, gold_table, pred_table, threshold):
    """
    Align tables based on the columns by constructing a score matrix
    """
    embeddings_gold = model.encode(list(gold_table.keys()))
    embeddings_pred = model.encode(list(pred_table.keys()))

    sim_mat_cosine = util.cos_sim(embeddings_gold, embeddings_pred).numpy()

    # calculate the alignment
    return _get_alignment(gold_table, pred_table, sim_mat_cosine, threshold)


def _get_alignment_columns_jaccard(gold_table, pred_table, threshold):
    """Constructs a score matrix of jaccard similarities"""
    score_matrix = np.array(
        [
            [jaccard_sentence(gold_colname, pred_colname) for pred_colname in pred_table.keys()]
            for gold_colname in gold_table.keys()
        ]
    )

    return _get_alignment(gold_table, pred_table, score_matrix, threshold)


def align_schema(target, prediction, method_name, sim_threshold, **config) -> tuple[dict, np.array]:
    """
    Externally avaiable method used for aligning predicted and gold target tables' schema (column names)

    config: other details about the method

    Returns a tuple. The first element is a dict whose keys are columns in the *target* table and whose
        values are a list of columns in the *prediction* table whose scores match. The second element is
        a 2-d np.array with shape of (len(target), len(prediction)) that contains the pairwise scores
        between each target and prediction column name
    """
    if "sentence_transformer" in method_name:
        # See: https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models
        # Fast but worse: all-MiniLM-L6-v2
        # Slow but better: all-mpnet-base-v2
        # not sure how these do with scientific text...
        model = SentenceTransformer(config.get("model", "all-MiniLM-L6-v2"))

        if method_name == "sentence_transformer-columns":
            return _get_alignment_columns_st(model, target, prediction, sim_threshold)
        elif method_name == "sentence_transformer-values":
            return _get_alignment_values_st(model, target, prediction, sim_threshold)
    elif method_name == "jaccard":
        return _get_alignment_columns_jaccard(target, prediciton, sim_threshold)


def decontextualize_table(table, metadata):
    # TODO: Implement this
    return table


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


def get_similar_sentence(query: str, pool: list[str], method="sentence_transformer") -> tuple[str, float]:
    """
    Returns the most similar string in the pool to the query along with the similarity scores.
    Uses
    """
    if method == "sentence_transformer":
        # See: https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models
        # Fast but worse: all-MiniLM-L6-v2
        # Slow but better: all-mpnet-base-v2
        # not sure how these do with scientific text...
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode([query] + pool, convert_tensors=True)
        scores = util.cos_sim(embeddings[0], embeddings[1:]).numpy()
        max_sent_idx = np.argmax(scores)
        return pool[max_sent_idx], scores[max_sent_idx]

    elif method in ("jaccard_keywords", "jaccard"):
        keywords = method == "jaccard_keywords"
        scores = [jaccard_sentence(query, sentence, keywords=keywords) for sentence in pool]
        max_sent_idx = np.argmax(scores)
        return pool[max_sent_idx], scores[max_sent_idx]

    else:
        raise ValueError(f"Invalid similarity method: {method}")
