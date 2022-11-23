HYDRA_CONFIG_PATH=~/acleto/jupyterlab_demo/configs \
HYDRA_CONFIG_NAME=al_ner \
python ~/acleto/acleto/al4nlp/utils/cache_all_necessary_files.py \
data.dataset_name=conll2003 acquisition_model.checkpoint=distilbert-base-cased cache_model_and_dataset=True cache_dir=~/acleto/jupyterlab_demo/cache/ner


HYDRA_CONFIG_PATH=~/acleto/jupyterlab_demo/configs \
HYDRA_CONFIG_NAME=al_cls \
python ~/acleto/acleto/al4nlp/utils/cache_all_necessary_files.py \
data.dataset_name=ag_news acquisition_model.checkpoint=distilbert-base-uncased cache_model_and_dataset=True cache_dir=~/acleto/jupyterlab_demo/cache/cls
