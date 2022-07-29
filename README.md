![img.png](img.png)
**ALToolbox**: a framework for computationally efficient and reusable active learning in NLP inside Jupyter Notebook
<hr>

[Installation](#installation) | [Quick Start](#quick_start) | [Docs](#documentation) | [Citation](#citation)

ALToolbox provides state-of-the-art **Active Learning** for Sequence Classification and Token Classification. 
Several pre-implemented Query Strategies, Initialization Strategies, and Stopping Criterion are provided, 
which can be easily mixed and matched to build active learning applications or run experiments.



## Features


## <a name="installation"></a>⚙️ Installation ⚙️

```bash
pip install -e ./active_learning
```


## <a name="quick_start"></a>💫 Quick Start 💫

For quick start, please see the examples of launching an active learning system or benchmarking a novel query stategy / unlabeled pool subsampling strategy for text classification and token classification tasks:

| # | Notebook                                                                   |
| --- |----------------------------------------------------------------------------|
| 1 | [Launching Active Learning for Text Classification](TODO: link to CLS AL)  |
| 2 | [Launching Active Learning for Token Classification](TODO: link to NER AL) |
| 3 | [Benchmarking a novel AL query strategy / unlabeled pool subsampling strategy](examples/benchmark_custom_strategy.ipynb)                        


## <a name="documentation"></a>📕 Documentation 📘



### 👨‍💻 Usage 👩‍💻
The `configs` folder contains config files with general settings. The `experiments` folder contains config files with experimental design. To run an experiment with a chosen configuration, specify config file name in `HYDRA_CONFIG_NAME` variable and run `train.sh` script (see `./examples/al` for details). 

For example to launch PLASM on AG-News with ELECTRA as a successor model:
```
cd PATH_TO_THIS_REPO
HYDRA_CONFIG_PATH=../experiments/ag_news HYDRA_EXP_CONFIG_NAME=ag_plasm python active_learning/run_tasks_on_multiple_gpus.py
```

### 👩‍🏫 Config structure explanation 📃
- `cuda_devices`: list of CUDA devices to use: one experiment on one CUDA device. `cuda_devices=[0,1]` means using zero-th and first devices.
- `config_name`: name of config from **configs** folder with general settings: dataset, experiment setting (e.g. LC/ASM/PLASM), model checkpoints, hyperparameters etc.
- `config_path`: path to config with general settings.
- `command`: **.py** file to run. For AL experiments, use **run_active_learning.py**.
- `args`: arguments to modify from a general config in the current experiment. `acquisition_model.name=xlnet-base-cased` means that _xlnet-base-cased_ will be used as an acquisition model.
- `seeds`: random seeds to use. `seeds=[4837, 23419]` means that two separate experiments with the same settings (except for **seed**) will be run: one with **seed == 4837**, one with **seed == 23419**.

### 👨‍🏫 Output explanation 🧾
By default, the results will be present in the folder `RUN_DIRECTORY/workdir_run_active_learning/DATE_OF_RUN/${TIME_OF_RUN}_${SEED}_${MODEL_CHECKPOINT}`. For instance, when launching from the repository folder: `al_nlp_feasible/workdir/run_active_learning/2022-06-11/15-59-31_23419_distilbert_base_uncased_bert_base_uncased`.

- When running a classic AL experiment (acquisition and successor models coincide, regardless of using UPS), the file with the model metrics is `acquisition_metrics.json`.
- When running an acquisition-successor mismatch experiment, the file with the model metrics is `successor_metrics.json`.
- When running a PLASM experiment, the file with the model metrics is `target_tracin_quantile_-1.0_metrics.json` (**-1.0** stands for the filtering value, meaning adaptive filtering rate; when using a deterministic filtering rate (e.g. **0.1**), the file will be named `target_tracin_quantile_0.1_metrics.json`). The file with the metrics of the model **without filtering** is `target_metrics.json`.

### 🆕️ New strategies addition 🎩
An AL query strategy should be designed as a function that:
   1) Receives 3 positional arguments and additional strategy kwargs:
     - `model` of class `TransformersBaseWrapper` or `WrapperEncoderPytorch` or `ModelFlairWrapper`: model wrapper;
     - `X_pool` of class `Dataset` or `TransformersDataset`: dataset with the unlabeled instances;
     - `n_instances` of class `int`: number of instances to query;
     - `kwargs`: additional strategy-specific arguments.
   2) Outputs 3 objects in the following order:
      - `query_idx` of class `array-like`: array with the indices of the queried instances;
      - `query` of class `Dataset` or `TransformersDataset`: dataset with the queried instances;
      - `uncertainty_estimates` of class `np.ndarray`: uncertainty estimates of the instances from `X_pool`. The higher the value - the more uncertain the model is in the instance.

The function with the strategy should be named the same as the file where it is placed (e.g. function `def my_strategy` inside a file `path_to_strategy/my_strategy.py`).
Use your strategy, setting `al.strategy=PATH_TO_FILE_YOUR_STRATEGY` in the experiment config.
The example is presented in `examples/Add_new_strategy.ipynb`

### 🆕️ New pool subsampling strategies addition 🎩
The addition of a new pool subsampling query strategy is similar to the addition of an AL query strategy. A subsampling strategy should be designed as a function that:
   1) It must receive 2 positional arguments and additional subsampling strategy kwargs:
     - `uncertainty_estimates` of class `np.ndarray`: uncertainty estimates of the instances in the order they are stored in the unlabeled data;
     - `gamma_or_k_confident_to_save` of class `float` or `int`: either a share / number of instances to save (as in random / naive subsampling) or an internal parameter (as in UPS);
     - `kwargs`: additional subsampling strategy specific arguments.
   2) It must output the indices of the instances to use (sampled indices) of class `np.ndarray`.

The function with the strategy should be named the same as the file where it is placed (e.g. function `def my_subsampling_strategy` inside a file `path_to_strategy/my_subsampling_strategy.py`).
Use your subsampling strategy, setting `al.sampling_type=PATH_TO_FILE_YOUR_SUBSAMPLING_STRATEGY` in the experiment config.
The example is presented in `examples/Add_new_strategy.ipynb`

### 📑 Datasets 📑
The research has employed 2 NER datasets (CoNLL-2003, OntoNotes-2012) and 2 Text Classification (CLS) datasets (AG-News, IMDB). If one wants to launch an experiment on a custom dataset, they need to use one of the following ways to add it:

1) Upload to [Hugging Face datasets](https://huggingface.co/datasets) and set: `config.data.path=datasets, config.data.dataset_name=DATASET_NAME, config.data.text_name=COLUMN_WITH_TEXT_OR_TOKENS_NAME, config.data.label_name=COLUMN_WITH_LABELS_OR_NER_TAGS_NAME`
2) Upload to **data/DATASET_NAME** folder, create **train.csv** / **train.json** file with the dataset, and set: `config.data.path=PATH_TO_THIS_REPO/data, config.data.dataset_name=DATASET_NAME, config.data.text_name=COLUMN_WITH_TEXT_OR_TOKENS_NAME, config.data.label_name=COLUMN_WITH_LABELS_OR_NER_TAGS_NAME`
3) \* Upload to **data/DATASET_NAME** **train.txt**, **dev.txt**, and **test.txt** files and set the arguments as in the previous point.
4) \*\* Upload to **data/DATASET_NAME** with each folder for each class, where each file in the folder contains a text with the label of the folder. For details, please see the **bbc_news** dataset in **./data**. The arguments must be set as in the previous two points.

\* - only for NER datasets

\*\* - only for CLS datasets

### 📟 Models 📟
The current version of the repository supports all models from [HuggingFace Transformers](https://huggingface.co/models), which can be used with `AutoModelForSequenceClassification` / `AutoModelForTokenClassification` classes (for CLS / NER). For CNN-based / BiLSTM-CRF models, please see the **al_cls_cnn.yaml** / **al_ner_bilstm_crf.yaml** configs from **./configs** folder for details.

### 🧑‍💻 Testing 🧪
By default, the tests will be run on the `cuda:0` device if CUDA is available or on CPU, otherwise. If one wants to manually specify the device for running the tests:

- On CPU: `CUDA_VISIBLE_DEVICES="" python -m pytest PATH_TO_REPO/tests`;
- On CUDA: `CUDA_VISIBLE_DEVICES="DEVICE_OR_DEVICES_NUMBER" python -m pytest PATH_TO_REPO/tests`.

We recommend to use CPU for the robustness of the results. The tests for CUDA are written under **Tesla V100-SXM3 32GB, CUDA V.10.1.243**. 

## 👯 Alternatives 👯‍♀️

[FAMIE](https://github.com/nlp-uoregon/famie), [Small-Text](https://github.com/webis-de/small-text), [modAL](https://github.com/modAL-python/modAL), [ALiPy](https://github.com/NUAA-AL/ALiPy), [libact](https://github.com/ntucllab/libact)

## <a name="citation"></a>💬 Citation 🗯

```

```

## 📄 License 📃
© 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research Institute" (AIRI). All rights reserved.

Licensed under the [MIT License](LICENSE).
