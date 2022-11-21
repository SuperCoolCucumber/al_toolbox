.. _installation:

============
Installation
============

You can easily install AlToolBox using pip:


.. code-block:: console

    pip install acleto

To annotate instances for active learning in Jupyter Notebook or Jupyter Lab one have to install additional widget after framework installation. In case of Jupyter Notebook usage run:

.. code-block:: console

    jupyter nbextension install --py --symlink --sys-prefix text_selector
    jupyter nbextension enable --py --sys-prefix text_selector

In case of Jupyter Lab usage run:

.. code-block:: console

    jupyter labextension install js
    jupyter labextension install text_selector