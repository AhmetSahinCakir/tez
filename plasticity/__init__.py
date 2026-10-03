"""Plasticity: bounded-periodic weight reparametrization for loss of plasticity in continual learning.

Experimental infrastructure for the thesis
"Derin sürekli öğrenmede plastisite kaybına karşı sınırlı-periyodik ağırlık yeniden parametrizasyonu".

Main entry points
-----------------
* ``python -m plasticity.run --config configs/pmnist_pilot.yaml --set method.name=sin seed=3``
* ``python scripts/run_suite.py --suite suites/pmnist_main.yaml`` (parallel grid of runs)
* ``python scripts/analyze.py --results results/pmnist_main`` (tables, statistics, figures)
"""

__version__ = "0.1.0"
