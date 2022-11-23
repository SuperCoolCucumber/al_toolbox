import gc
import time
from typing import Union, Tuple, List
import os
import numpy as np
import logging
from datasets.metric import Metric

from torch import cuda

from .transformers_base_wrapper import TransformersBaseWrapper
from .wrapper_seq2seq import WrapperSeq2Seq
from ...utils.seq2seq_metrics import decode_all, calculate_bleu, calculate_additional_hf_metrics
from ...utils.summarization_metrics import (
    prepare_data_for_abssum_metrics,
    calculate_rouge,
    calculate_summac_score,
    calculate_bartscore,
    calculate_abstractiveness,
)


log = logging.getLogger()
USE_ADD_ATS_METRICS = os.environ.get("USE_ADD_ATS_METRICS", True)

if USE_ADD_ATS_METRICS:
    try:
        from ...utils.summarization_metrics import BARTScorer, SummaCZS

        ADD_METRICS_IMPORTED = True
    except Exception as e:
        log.info(f"Cannot import BARTScore & SummacZS. Exception:\n{e}")
        ADD_METRICS_IMPORTED = False


class WrapperAts(WrapperSeq2Seq, TransformersBaseWrapper):
    @staticmethod
    def get_tokenize_function(
        tokenizer, text_name, label_name, test_mode: bool = False, **kwargs
    ):
        if test_mode:

            def tokenize_function(instances):
                encoded = tokenizer(instances[text_name], truncation=True)
                return encoded

        else:

            def tokenize_function(instances):
                encoded = tokenizer(instances[text_name], truncation=True)
                if not test_mode:
                    with tokenizer.as_target_tokenizer():
                        labels = tokenizer(instances[label_name], truncation=True)
                    encoded["labels"] = labels["input_ids"]
                return encoded

        return tokenize_function

    @staticmethod
    def get_metric_name():
        return "test_rouge1"

    @staticmethod
    def get_datasets_metric_name():
        return "rouge"

    @staticmethod
    def get_additional_datasets_metric_names():
        return ["sacrebleu"]

    def get_kwargs_for_compute_metrics_fn(self):
        return {
            "tokenizer": self.tokenizer,
            "use_additional_metrics": USE_ADD_ATS_METRICS,
        }

    @staticmethod
    def get_compute_metrics_function(
        metric,
        tokenizer,
        additional_metrics: Union[Tuple[Metric], List[Metric]] = tuple(),
        use_additional_metrics: bool = True,
        **bartscorer_init_params,
    ):
        assert tokenizer is not None, "Please provide tokenizer"

        def compute_metrics(eval_preds):
            decoded_preds, decoded_labels, decoded_texts = decode_all(
                eval_preds, tokenizer, use_additional_metrics=use_additional_metrics
            )
            (
                predictions,
                labels,
                is_zeroword,
                is_uniword,
                tokenized,
            ) = prepare_data_for_abssum_metrics(
                decoded_preds, decoded_labels, decoded_texts
            )
            # ROUGE metrics
            result = calculate_rouge(predictions, labels, is_zeroword, is_uniword)
            # Add BLEU metric
            bleu_scores = calculate_bleu(decoded_preds, decoded_labels)
            result.update(bleu_scores)
            # Add mean generated length
            prediction_lens = [
                np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
            ]
            result["gen_len"] = np.mean(prediction_lens)

            if use_additional_metrics and decoded_texts is not None:
                abstr_scores = calculate_abstractiveness(
                    tokenized["preds"], tokenized["labels"], tokenized["texts"]
                )
                result.update(abstr_scores)
                if ADD_METRICS_IMPORTED:
                    # BARTScore
                    scorer = BARTScorer(**bartscorer_init_params)
                    # probable error with texts
                    bart_scores = calculate_bartscore(
                        decoded_preds, decoded_labels, decoded_texts, scorer
                    )
                    del scorer
                    cuda.empty_cache()
                    gc.collect()
                    # SummaC-ZC score
                    scorer = SummaCZS(granularity="sentence", model_name="vitc")
                    summac_scores = calculate_summac_score(
                        decoded_preds, decoded_labels, decoded_texts, scorer
                    )
                    result.update(bart_scores)
                    result.update(summac_scores)

            add_result = calculate_additional_hf_metrics(additional_metrics, decoded_labels, decoded_preds)
            result.update(add_result)

            return result

        return compute_metrics
