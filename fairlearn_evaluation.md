# Python Library Evaluation: Fairlearn



## 1 Setup
#### 1.1 0.70 stable version
```python
#installed with pip from PyPI 
pip install fairlearn
#or on conda-forge
conda install -c conda-forge fairlearn
```
#### 1.2 main branch version
Requires manuel installation. 
```python
git clone git@github.com:fairlearn/fairlearn.git
## To install in editable mode using pip run
pip install -e .
```
Note: Fairlearn is subject to change, so notebooks downloaded from main may not be compatible with Fairlearn installed with pip.

## 2 Dataset
This module contains datasets that can be used for benchmarking and education.
```python
# UCI Adult dataset (binary classification)
# predict whether a person makes over $50,000 a year
from fairlearn.datasets import fetch_adult
# UCI bank marketing dataset (binary classification).
# predict if the client will subscribe a term deposit
from fairlearn.datasets import fetch_bank_marketing
# boston housing dataset (regression)
# There’s a “lower status of population” (LSTAT) parameter that you need to look out for 
# and a column that is a derived from the proportion of people with a black skin color that live in a neighborhood
from fairlearn.datasets import fetch_boston
```
Using fetch_adult dataset as exmaple:
```python
```


## 3 Mitigation

## 4 Metrics
