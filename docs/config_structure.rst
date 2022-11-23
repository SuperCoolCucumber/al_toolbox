.. _config_structure:

==========================
Config structure explained
==========================

- `cuda_devices`: list of CUDA devices to use: one experiment on one CUDA device. `cuda_devices=[0,1]` means using zero-th and first devices.
- `config_name`: name of config from **configs** folder with general settings: dataset, experiment setting (e.g. LC/ASM/PLASM), model checkpoints, hyperparameters etc.
- `config_path`: path to config with general settings.
- `command`: **.py** file to run. For AL experiments, use **run_active_learning.py**.
- `args`: arguments to modify from a general config in the current experiment. `acquisition_model.name=xlnet-base-cased` means that _xlnet-base-cased_ will be used as an acquisition model.
- `seeds`: random seeds to use. `seeds=[4837, 23419]` means that two separate experiments with the same settings (except for **seed**) will be run: one with **seed == 4837**, one with **seed == 23419**.
