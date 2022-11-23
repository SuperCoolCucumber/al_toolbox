.. _basic_usage:

================
Usage
================

The `configs` folder contains config files with general settings. The `experiments` folder contains config files with experimental design. To run an experiment with a chosen configuration, specify config file name in `HYDRA_CONFIG_NAME` variable and run `train.sh` script (see `./examples/al` for details).

For example to launch PLASM on AG-News with ELECTRA as a successor model:


.. code-block:: console

    cd PATH_TO_THIS_REPO
    HYDRA_CONFIG_PATH=../experiments/ag_news HYDRA_EXP_CONFIG_NAME=ag_plasm python active_learning/run_tasks_on_multiple_gpus.py
