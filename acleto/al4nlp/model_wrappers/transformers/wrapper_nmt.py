from typing import Union, Tuple, List
import numpy as np
import logging
from datasets.metric import Metric

from .transformers_base_wrapper import TransformersBaseWrapper
from .wrapper_seq2seq import WrapperSeq2Seq
from ...utils.seq2seq_metrics import decode_all, calculate_bleu, calculate_additional_hf_metrics


log = logging.getLogger()


class WrapperNmt(WrapperSeq2Seq, TransformersBaseWrapper):
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
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(instances[label_name], truncation=True)
                    encoded["labels"] = labels["input_ids"]
                return encoded

        return tokenize_function

    @staticmethod
    def get_metric_name():
        return "test_bleu"

    @staticmethod
    def get_datasets_metric_name():
        return "sacrebleu"

    @staticmethod
    def get_additional_datasets_metric_names():
        return []

    def get_kwargs_for_compute_metrics_fn(self):
        return {
            "tokenizer": self.tokenizer,
        }

    @staticmethod
    def get_compute_metrics_function(
        metric,
        tokenizer,
        additional_metrics: Union[Tuple[Metric], List[Metric]] = tuple(),
    ):
        assert tokenizer is not None, "Please provide tokenizer"

        def compute_metrics(eval_preds):
            decoded_preds, decoded_labels, decoded_texts = decode_all(
                eval_preds, tokenizer, use_additional_metrics=False
            )
            # Main metric
            result = calculate_bleu(decoded_preds, decoded_labels)
            # Add mean generated length
            prediction_lens = [
                np.count_nonzero(pred != tokenizer.pad_token_id)
                for pred in decoded_preds
            ]
            result["gen_len"] = np.mean(prediction_lens)
            # Sacrebleu
            sacrebleu_result = metric.compute(
                predictions=decoded_preds, references=[[lab] for lab in decoded_labels],
            )
            sacrebleu_result["sacrebleu"] = sacrebleu_result.pop("score")
            result.update(sacrebleu_result)

            add_result = calculate_additional_hf_metrics(additional_metrics, decoded_labels, decoded_preds)
            result.update(add_result)

            return result

        return compute_metrics
