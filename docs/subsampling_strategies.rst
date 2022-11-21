.. _subsampling_strategies:

======================
Subsampling Strategies
======================

For large datasets, making predictions for the
whole unlabeled set on each iteration to obtain the
uncertainty estimates may require an enormous
amount of time and resources.  Unlabeled pool subsampling algorithms are adressing this issue by
subsampling instances in the unlabeled pool depending on their uncertainty scores
obtained on previous AL iterations. This helps
to speed up the AL iterations, especially
when the unlabeled pool is large.

+-----+------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------+
| #   | Strategy                                                                                                                           | Citation                                                       |
+-----+------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------+
| 1   | `UPS <https://github.com/AIRI-Institute/al_toolbox/blob/main/acleto/al4nlp/pool_subsampling_strategies/ups_subsampling.py>`_       | `Citation <https://aclanthology.org/2022.findings-naacl.90/>`_ |
+-----+------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------+
| 2   | `Naïve <https://github.com/AIRI-Institute/al_toolbox/blob/main/acleto/al4nlp/pool_subsampling_strategies/naive_subsampling.py>`_   | `Citation <https://aclanthology.org/2022.findings-naacl.90/>`_ |
+-----+------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------+
| 3   | `Random <https://github.com/AIRI-Institute/al_toolbox/blob/main/acleto/al4nlp/pool_subsampling_strategies/random_subsampling.py>`_ | \-                                                             |
+-----+------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------+