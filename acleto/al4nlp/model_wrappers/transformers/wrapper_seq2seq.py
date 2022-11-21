import gc
import time
from tqdm import tqdm
from typing import Dict
import numpy as np
from abc import abstractmethod
from pathlib import Path
from requests.models import HTTPError

import torch
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from .transformers_base_wrapper import TransformersBaseWrapper
from ...utils.general import tensor_to_numpy

import logging

log = logging.getLogger()


class WrapperSeq2Seq(TransformersBaseWrapper):
    @staticmethod
    def get_model_class():
        return AutoModelForSeq2SeqLM

    @staticmethod
    def get_training_args_class():
        return Seq2SeqTrainingArguments

    @staticmethod
    def get_data_collator_class():
        return DataCollatorForSeq2Seq

    @staticmethod
    def get_trainer_class():
        return Seq2SeqTrainer

    def get_additional_training_kwargs(self, train_data):
        generation_max_length = (
            self._get_generation_max_length(train_data, self.data_config)
            if self._trainer_kwargs.generation_max_length is None
            else self._trainer_kwargs.generation_max_length
        )
        log.info(f"Using generation max length == {generation_max_length}")
        generation_num_beams = self._trainer_kwargs.generation_num_beams
        return {
            "predict_with_generate": True,
            "include_inputs_for_metrics": True,
            "generation_max_length": generation_max_length,
            "generation_num_beams": generation_num_beams,
        }

    def get_model_and_tokenizer(
        self, model_cfg, seed: int = 42, cache_dir=None, **kwargs
    ):
        set_seed(seed)
        pretrained_model_name = model_cfg.checkpoint

        model_cache_dir = Path(cache_dir) / "model" if cache_dir is not None else None
        tokenizer_cache_dir = (
            Path(cache_dir) / "tokenizer" if cache_dir is not None else None
        )
        ### Model part
        model_class = self.get_model_class()
        try:
            model = model_class.from_pretrained(
                pretrained_model_name, cache_dir=model_cache_dir,
            )
        # If there are troubles with the connection, try to build from local files
        except HTTPError:
            log.warning(
                f"Connection troubles. Using local model {pretrained_model_name}."
            )
            model = model_class.from_pretrained(
                pretrained_model_name, cache_dir=model_cache_dir, local_files_only=True,
            )
        ### Tokenizer part
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name, cache_dir=tokenizer_cache_dir
            )
        except HTTPError:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name,
                cache_dir=tokenizer_cache_dir,
                local_files_only=True,
            )
        if model_cfg.tokenizer_max_length is not None:
            tokenizer.model_max_length = model_cfg.tokenizer_max_length
        return model, tokenizer

    def get_predictions(
        self,
        data,
        is_tokenized=False,
        data_config=None,
        calculate_time=False,
        evaluate: bool = False,
        use_predict_loop: bool = False,
    ):
        if data_config is None:
            data_config = self.data_config
        text_name = data_config["text_name"]
        label_name = data_config["label_name"]

        if not is_tokenized:
            data = self.tokenize_data(
                tokenizer=self.tokenizer,
                data=data,
                text_name=text_name,
                label_name=label_name,
                test_mode=not evaluate,
            )
        assert (
            getattr(self, "trainer", None) is not None
        ), "Instantiate the Trainer first!"
        start_time = time.time()
        predictions = self.trainer.predict(data)

        if calculate_time:
            self._calculate_time(start_time, phase="predict")
        torch.cuda.empty_cache()
        gc.collect()

        return predictions

    def generate(
        self,
        data,
        return_scores: bool = True,
        return_decoded_preds: bool = True,
        is_tokenized: bool = False,
        data_config=None,
        to_numpy: bool = False,
        to_eval_mode: bool = True,
        batch_size: int = None,
        calculate_time: bool = True,
        generation_max_length: int = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Only implemented for seq2seq models (up to 05.09 for Abs-sum & NMT)
        :return: sequences of ids of most probable tokens & scores of each sequence (sum of log probs)
        """
        if data_config is None:
            data_config = self.data_config

        if not is_tokenized:
            data = self.tokenize_data(
                data=data,
                tokenizer=self.tokenizer,
                text_name=data_config["text_name"],
                label_name=data_config["label_name"],
            )
            if "labels" in data[0]:
                data = data.remove_columns(["labels"])

        start_time = time.time()
        if getattr(self, "trainer", None) is not None:
            log.warning("Model is not fine-tuned! Be careful applying it to new data.")
        output = self._model_generate_loop(
            data, batch_size, to_eval_mode, generation_max_length, **kwargs
        )

        if calculate_time:
            self._calculate_time(start_time, phase="predict")

        if to_numpy:
            output = {k: tensor_to_numpy(v) for k, v in output.items()}

        if return_decoded_preds:
            output["predictions"] = self.tokenizer.batch_decode(
                output["sequences"], skip_special_tokens=True
            )
            if kwargs.get("num_return_sequences", 1) > 1:
                num_ret_seq = kwargs.get("num_return_sequences", 1)
                preds = output["predictions"]
                output["predictions"] = [
                    preds[i * num_ret_seq : (i + 1) * num_ret_seq]
                    for i in range(len(preds) // num_ret_seq)
                ]
                output["sequences"] = output["sequences"].reshape(
                    -1, num_ret_seq, output["sequences"].shape[1]
                )
                output["sequences_scores"] = output["sequences_scores"].reshape(
                    -1, num_ret_seq
                )

        if not return_scores:
            output.pop("sequences_scores")
        torch.cuda.empty_cache()
        gc.collect()

        return output

    def _model_generate_loop(
        self,
        data,
        batch_size=None,
        to_eval_mode=True,
        generation_max_length: int = None,
        **kwargs,
    ) -> Dict:
        """
        Returns: dict with sequences of ids and their scores
        """
        torch.cuda.empty_cache()
        if batch_size is None:
            batch_size = (
                self._batch_size_kwargs.get("generate_batch_size", None)
                or self._batch_size_kwargs.eval_batch_size
            )

        dataloader = DataLoader(
            data,
            batch_size=batch_size,
            collate_fn=DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer, padding="longest"
            ),
        )

        if generation_max_length is not None:
            max_length = sequences_length = generation_max_length
        elif getattr(self, "trainer", None) is not None:
            max_length = sequences_length = self.trainer.args.generation_max_length
        else:
            max_length = None
            sequences_length = 128
        num_return_sequences = kwargs.get("num_return_sequences", 1)

        if to_eval_mode:
            self.model.eval()
        device = self.model.device
        sequences_X_shape = num_return_sequences * len(data)
        sequences = (
            torch.zeros(
                (sequences_X_shape, sequences_length), dtype=torch.int64, device=device
            )
            + self.tokenizer.pad_token_id
        )
        scores = torch.empty(sequences_X_shape, dtype=torch.float32, device=device)

        with torch.no_grad():
            start = 0
            for batch in tqdm(dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                output = self.model.generate(
                    **batch,
                    max_length=max_length,
                    min_length=3,  # To avoid empty summaries. 3 == <BOS> + at least one token + <EOS>
                    output_scores=True,
                    return_dict_in_generate=True,
                    **kwargs,
                )
                end = start + (len(batch["input_ids"]) * num_return_sequences)
                sequences[start:end, : output.sequences.shape[1]].copy_(
                    output.sequences, non_blocking=True
                )
                scores[start:end].copy_(output.sequences_scores, non_blocking=True)
                start = end

        return {"sequences": sequences, "sequences_scores": scores}

    def _get_generation_max_length(self, data=None, data_config=None):
        if data is None:
            data = self.tokenized_dev_data
        if data is None:
            data_config = data_config if data_config is not None else self.data_config
            data = self.tokenize_data(
                tokenizer=self.tokenizer,
                data=self.dev_data,
                text_name=data_config["text_name"],
                label_name=data_config["label_name"],
            )
            self.tokenized_dev_data = data
        gen_max_length = min(
            # Using quantile instead of `max` since `max` can lead to enormously long generation
            round(np.quantile([len(x) for x in data["labels"]], 0.95) + 1),
            self.tokenizer.model_max_length,
        )
        return gen_max_length
