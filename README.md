# PySATL-CPD-Module

<a href="https://github.com/Lesh79/PySATL-CPD-Module/actions"><img alt="Test" src="https://github.com/Lesh79/PySATL-CPD-Module/actions/workflows/check.yaml/badge.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://github.com/Lesh79/PySATL-CPD-Module/blob/main/LICENSE"><img alt="License: MIT" src="https://black.readthedocs.io/en/stable/_static/license.svg"></a>

**Change point detection module** (*abbreviated CPD module*) is a module, designed for detecting anomalies in time series data, which refer to significant deviations from expected patterns or trends. Anomalies can indicate unusual events or changes in a system, making them crucial for monitoring and analysis in various fields such as finance, healthcare, and network security.

At the moment, the module implements the following CPD algorithms:
* Bayesian algorithm
* Density based algorithms:
    * KLIEP
    * RuLSIF
* Graph algorithm
* k-NN based algorithm

---

## Requirements

- Python 3.10+
- numpy 2.0.0+
- scipy 1.14.0+
- matplotlib 3.9.1+

## Installation

Clone the repository:

```bash
git clone https://github.com/Lesh79/PySATL-CPD-Module
```

Install dependencies:

```bash
poetry install
```

## CPD module usage example:

```python
#import needed CPD algorithm from CPDShell.Core
from CPDShell.Core.algorithms.graph_algorithm import GraphAlgorithm
from CPDShell.labeled_data import LabeledCPData

#import shell
from CPDShell.shell import CPDShell


# make a shell object
shell = CPDShell([1] * 100 + [50]*100 + [100] * 100)

# specify CPD algorithm with parametrs
shell.CPDalgorithm = GraphAlgorithm(lambda a, b: abs(a - b) < 5, 3)

# then run algorithm
change_points = shell.run_cpd()

# print the results
print(change_points)
# output:
# Located change points: (100;200)
# Computation time (ms): 0.03

# visualize data with located changepoints
change_points.visualize()
```
![example_of_output](assets/exam1.png)

## Development

Install requirements

```bash
poetry install --with dev
```

## Pre-commit

Install pre-commit hooks:

```shell
poetry run pre-commit install
```

Starting manually:

```shell
poetry run pre-commit run --all-files --color always --verbose --show-diff-on-failure
```

## License

This project is licensed under the terms of the **MIT** license. See the [LICENSE](LICENSE) for more information.
