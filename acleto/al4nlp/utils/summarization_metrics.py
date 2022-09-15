import numpy as np
from typing import List, Tuple, Dict, Union
import time
from datasets import load_metric
import os
from pathlib import Path
import sys
import logging

from rouge_score import tokenize
from nltk.stem import porter
from nltk import ngrams
from nltk.tokenize import sent_tokenize

log = logging.getLogger()

if os.environ.get("USE_ADD_ATS_METRICS", True):
    try:
        path = Path("/".join(os.path.abspath(__file__).split("/")[:-1]))
        sys.path.append(str(path / "../../utils/packages/summac"))
        sys.path.append(str(path / "../../utils/packages/BARTScore"))
        from .packages.bart_score import BARTScorer
        from .packages.summac.summac.model_summac import SummaCZS

        log.info("Successfully loaded BARTScore and SummaC models.")

        ADD_METRICS_IMPORTED = True
    except:
        log.info(
            "Cannot load BARTScore & SummaC models;"
            " Consistency scores will not be computed for abstractive summarization."
        )
        ADD_METRICS_IMPORTED = False


def calculate_ngram_overlap(summary, text, n=1, use_modified=True):
    summary_ngrams = list(ngrams(summary, n))
    text_ngrams = list(ngrams(text, n))

    if len(summary_ngrams) > 0:
        ngrams_intersection = set(summary_ngrams).intersection(set(text_ngrams))
        if use_modified:
            word_is_part_of_ngram_copied = [
                any((x in ngram for ngram in ngrams_intersection)) for x in summary
            ]
            return 1 - sum(word_is_part_of_ngram_copied) / len(
                word_is_part_of_ngram_copied
            )
        else:
            return sum([x not in ngrams_intersection for x in summary_ngrams]) / len(
                summary_ngrams
            )
    return None


def calculate_rouge(
    predictions: List[str],
    labels: List[str],
    is_zeroword: np.ndarray = None,
    is_uniword: np.ndarray = None,
) -> Dict[str, np.ndarray]:
    """
    Calculate ROUGE scores
    :param predictions:
    :param labels:
    :param is_zeroword:
    :param is_uniword:
    :return: dictionary with 4 values
    """
    if is_zeroword is None:
        is_zeroword = np.zeros(len(predictions), dtype=bool)
    if is_uniword is None:
        is_uniword = np.zeros(len(predictions), dtype=bool)

    rouge = load_metric("rouge")
    rouges = rouge.compute(
        predictions=predictions,
        references=labels,
        use_stemmer=True,
        use_aggregator=False,
    )
    metrics = np.array([[x.fmeasure for x in value] for value in rouges.values()])
    # Substitute invalid observations with nans
    rouge1 = np.mean(metrics[0][~is_zeroword])
    rouge2 = np.mean(metrics[1][~(is_zeroword | is_uniword)])
    rougeL = np.mean(metrics[2][~is_zeroword])
    rougeLsum = np.mean(metrics[3][~is_zeroword])
    metrics = {
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL,
        "rougeLsum": rougeLsum,
    }
    return metrics


def calculate_summac_score(
    predictions: List[str], labels: List[str], texts: List[str], scorer=None
) -> Dict[str, np.ndarray]:
    start_time = time.time()
    if scorer is None:
        scorer = SummaCZS(granularity="sentence", model_name="vitc")
    preds_score = np.mean(scorer.score(texts, predictions)["scores"])
    labels_score = np.mean(scorer.score(texts, labels)["scores"])
    log.info(f"SummaC took {time.time() - start_time:.4f} seconds")
    return {"SUMMAC-pred": preds_score, "SUMMAC-label": labels_score}


def calculate_bartscore(
    predictions, labels, texts, scorer=None, batch_size=4, **bartscorer_init_params
) -> Dict[str, np.ndarray]:

    decoded_texts = ["\n".join(sent_tokenize(text.strip())) for text in texts]
    start_time = time.time()
    if scorer is None:
        scorer = BARTScorer(**bartscorer_init_params)
    scores = get_bart_scores(
        scorer,
        preds=predictions,
        refs=labels,
        texts=decoded_texts,
        batch_size=batch_size,
        aggregate=True,
    )
    log.info(f"BARTScore took {time.time() - start_time:.4f} seconds")
    return scores


def get_bart_scores(scorer, preds, refs, texts, batch_size=4, aggregate=True):
    scores = {}
    scores["BARTScore-sh"] = np.array(scorer.score(texts, preds, batch_size=batch_size))
    scores["BARTScore-rh"] = np.array(scorer.score(refs, preds, batch_size=batch_size))
    scores["BARTScore-hr"] = np.array(scorer.score(preds, refs, batch_size=batch_size))
    scores["BARTScore-f1"] = (
        2
        * scores["BARTScore-rh"]
        * scores["BARTScore-hr"]
        / (scores["BARTScore-rh"] + scores["BARTScore-hr"])
    )
    scores["BARTScore-fa"] = (scores["BARTScore-rh"] + scores["BARTScore-hr"]) / 2

    if aggregate:
        scores = {key: np.mean(value) for key, value in scores.items()}
    return scores


def prepare_data_for_abssum_metrics(
    predictions: List[str], labels: List[str], texts: Union[List[str], None] = None
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    predictions = ["\n".join(sent_tokenize(pred)) for pred in predictions]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]
    # Calculate zero- and uni-word summaries (that must be excluded from calculating
    # ROUGE-1 and ROUGE-L for the former and ROUGE-2 for both)
    stemmer = porter.PorterStemmer()
    tokenized_labels = [tokenize.tokenize(x, stemmer) for x in labels]
    len_tokenized_labels = np.array([len(x) for x in tokenized_labels])
    is_uniword = len_tokenized_labels == 1
    is_zeroword = len_tokenized_labels == 0

    if texts is not None:
        tokenized = {}
        tokenized["preds"] = [tokenize.tokenize(x, stemmer) for x in predictions]
        tokenized["texts"] = [tokenize.tokenize(x, stemmer) for x in texts]
        tokenized["labels"] = tokenized_labels
    else:
        tokenized = None
    return predictions, labels, is_zeroword, is_uniword, tokenized


def calculate_abstractiveness(
    tokenized_preds: List[List[str]],
    tokenized_labels: List[List[str]],
    tokenized_texts: List[List[str]],
) -> Dict[str, float]:
    result = {}
    for use_modified in [False, True]:
        for n in range(1, 5):
            pred_ngram_overlaps = []
            label_ngram_overlaps = []
            for pred, label, text in zip(
                tokenized_preds, tokenized_labels, tokenized_texts
            ):
                pred_pair_ngram_overlap = calculate_ngram_overlap(
                    pred, text, n, use_modified
                )
                if pred_pair_ngram_overlap is not None:
                    pred_ngram_overlaps.append(pred_pair_ngram_overlap)
                label_pair_ngram_overlap = calculate_ngram_overlap(
                    label, text, n, use_modified
                )
                if label_pair_ngram_overlap is not None:
                    label_ngram_overlaps.append(label_pair_ngram_overlap)
            key = f"ngram_overlap_{n}" if use_modified else f"novel_ngrams_{n}"
            result["pred_" + key] = (
                np.mean(pred_ngram_overlaps) if len(pred_ngram_overlaps) > 0 else -1
            )
            result["label_" + key] = (
                np.mean(label_ngram_overlaps) if len(label_ngram_overlaps) > 0 else -1
            )
    return result
