import os
from pathlib import Path
import argparse


if (
    Path("/proc/driver/nvidia/version").exists()
    and os.environ.get("USE_CUDA_FOR_TESTS", True)
    and len(os.environ.get("CUDA_VISIBLE_DEVICES", "0")) > 0
):
    USE_CUDA = True
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_ID_TO_USE", "0")
else:
    USE_CUDA = False
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


MODEL_OVERRIDES = lambda model_type: [
    f"{model_type}.training.trainer_args.fp16.training={USE_CUDA}",
    f"{model_type}.training.batch_size_args.min_num_gradient_steps=10",
    f"{model_type}.training.trainer_args.num_epochs=1",
    f"{model_type}.training.trainer_args.serialization_dir=output",
    f"{model_type}.training.batch_size_args.adjust_num_epochs=False",
    f"{model_type}.training.batch_size_args.adjust_batch_size=False",
]

AL_OVERRIDES = [
    "al.num_queries=1",
    "seed=42",
    "al.iters_to_recalc_scores=no",
    "al.sampling_type=random",
    "hydra.run.dir=.",
]

FULL_DATA_OVERRIDES = [
    "seed=42",
    "hydra.run.dir=.",
]

DA_OVERRIDES = [
    "gpus=[0]",
    "weight_decay=1e-3",
    "learning_rate=1e-6",
    "gradient_accumulation_steps=4",
    "n_gpu=1",
    "max_steps=1",
    "hydra.run.dir=.",
]

DA_NER_OVERRIDES = DA_OVERRIDES + [
    "data.dataset_name=ner_test",
    f"output_dir=test_da_ner_use_{'cuda' if USE_CUDA else 'cpu'}/",
    "train_data_file=./train_ner_test.txt",
    "eval_data_file=./dev_ner_test.txt",
    "data.source_task=ner",
    "data.text_name=tokens",
    "data.label_name=tags",
]

DA_CLS_OVERRIDES = DA_OVERRIDES + [
    "data.dataset_name=bbc_news",
    f"output_dir=test_da_cls_use_{'cuda' if USE_CUDA else 'cpu'}/",
    "train_data_file=./train_bbc_news.txt",
    "eval_data_file=./dev_bbc_news.txt",
    "data.source_task=cls",
    "data.text_name=text",
    "data.label_name=label",
]

NER_TRANSFORMERS = [
    "data.dataset_name=ner_test",
    "+data.use_subset=0.01",
    f"output_dir=test_ner_use_{'cuda' if USE_CUDA else 'cpu'}/",
    "al.step_p_or_n=0.01",
    "al.init_p_or_n=0.01",
    "al.gamma_or_k_confident_to_save=0.1",
]

CLS_TRANSFORMERS = [
    "data.dataset_name=bbc_news",
    "+data.use_subset=0.01",
    "data.train_size_split=0.95",
    f"output_dir=test_cls_use_{'cuda' if USE_CUDA else 'cpu'}/",
    "al.step_p_or_n=0.002",
    "al.init_p_or_n=0.002",
    "al.gamma_or_k_confident_to_save=0.02",
]

ATS_TRANSFORMERS = [
    "data.dataset_name=abssum_debate",
    "data.text_name=argument_sentences",
    "data.label_name=claim",
    "+data.use_subset=0.01",
    f"output_dir=test_ner_use_{'cuda' if USE_CUDA else 'cpu'}/",
    "al.step_p_or_n=1",
    "al.init_p_or_n=1",
    "al.gamma_or_k_confident_to_save=10",
]

AL_NER_TRANSFORMERS = (
    AL_OVERRIDES + MODEL_OVERRIDES("acquisition_model") + NER_TRANSFORMERS
)
AL_CLS_TRANSFORMERS = (
    AL_OVERRIDES + MODEL_OVERRIDES("acquisition_model") + CLS_TRANSFORMERS
)
AL_ATS_TRANSFORMERS = (
    AL_OVERRIDES + MODEL_OVERRIDES("acquisition_model") + ATS_TRANSFORMERS +\
    ["acquisition_model.checkpoint=t5-small",
     "acquisition_model.tokenizer_max_length=16"]
)
FULL_NER_TRANSFORMERS = (
    FULL_DATA_OVERRIDES + MODEL_OVERRIDES("model") + NER_TRANSFORMERS[:3]
)
FULL_CLS_TRANSFORMERS = (
    FULL_DATA_OVERRIDES + MODEL_OVERRIDES("model") + CLS_TRANSFORMERS[:4]
)
FULL_ATS_TRANSFORMERS = (
    FULL_DATA_OVERRIDES + MODEL_OVERRIDES("model") + ATS_TRANSFORMERS[:5] +\
    ["model.training.trainer_args.label_smoothing_factor=0.0",
     "model.tokenizer_max_length=16",
     "model.checkpoint=t5-small"]
)
SUCCESSOR_OVERRIDES = MODEL_OVERRIDES("successor_model")
TARGET_OVERRIDES = MODEL_OVERRIDES("target_model")

DIST_BINARIZE_NAMESPACE = argparse.Namespace(dataset_name="trec",
                                             tokenizer_type="electra",
                                             tokenizer_name="google/electra-base-discriminator",
                                             dump_file="./data/binarized_text",
                                             cache_dir="./data")
DIST_COUNTS_NAMESPACE = argparse.Namespace(data_file="data/binarized_text.electra.pickle",
                                           token_counts_dump="data/token_counts.electra.pickle",
                                           vocab_size=30522)
DIST_TRAIN_NAMESPACE = argparse.Namespace(student_type="electra",
                                          student_config="../../../distillation/training_configs/distilelectra.json",
                                          teacher_type="electra",
                                          teacher_name="google/electra-base-discriminator",
                                          alpha_ce=5.0, alpha_mlm=2.0, alpha_cos=1.0,
                                          alpha_act=1.0, alpha_clm=0.0, n_epoch=1,
                                          mlm=True, freeze_pos_embs=True, force=True,
                                          data_file="data/binarized_text.electra.pickle",
                                          token_counts="data/token_counts.electra.pickle",
                                          dump_path="./serialization_dir/distilelectra",
                                          student_pretrained_weights=None,
                                          freeze_token_type_embds=False,
                                          temperature=2.0, alpha_mse=0.0,
                                          mlm_mask_prop=0.15, word_mask=0.8,
                                          word_keep=0.1, word_rand=0.1, mlm_smoothing=0.7,
                                          restrict_ce_to_mask=False, batch_size=5,
                                          group_by_size=True, gradient_accumulation_steps=50,
                                          warmup_prop=0.05, weight_decay=0.0,
                                          learning_rate=1e-4, adam_epsilon=1e-6,
                                          max_grad_norm=5.0, initializer_range=0.02,
                                          fp16=False, fp16_opt_level="O1",
                                          n_gpu=1, local_rank=-1, seed=56,
                                          log_interval=500, checkpoint_interval=4000)
