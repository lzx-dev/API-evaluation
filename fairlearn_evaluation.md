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
data = fetch_adult(as_frame=True, return_X_y=False)
# pandas DataFrame including columns with appropriate dtypes (numeric, string or categorical)
x = data.data
# pandas DataFrame or Series depending on the number of target_columns
y = data.target 
# Array of ordered feature names used in the dataset
feature_names = data.feature_names
# Description of the UCI Adult dataset
descr = data.DESCR

# If return_X_y is True, returns (data.data, data.target) instead of a Bunch object.
data_2 = fetch_adult(as_frame=True, return_X_y=True)
x, y = data_2

## if as_frame is False, return type change from pandas.DataFrame to numpy.ndarray
```
Output for data.DESCR
```**Author**: Ronny Kohavi and Barry Becker  
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/Adult) - 1996  
**Please cite**: Ron Kohavi, "Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid", Proceedings of the Second International Conference on Knowledge Discovery and Data Mining, 1996  
Prediction task is to determine whether a person makes over 50K a year. Extraction was done by Barry Becker from the 1994 Census database. A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
This is the original version from the UCI repository, with training and test sets merged.

### Variable description
Variables are all self-explanatory except __fnlwgt__. This is a proxy for the demographic background of the people: "People with similar demographic characteristics should have similar weights". This similarity-statement is not transferable across the 51 different states.

Description from the donor of the database: 

The weights on the CPS files are controlled to independent estimates of the civilian noninstitutional population of the US.  These are prepared monthly for us by Population Division here at the Census Bureau. We use 3 sets of controls. These are:
1.  A single cell estimate of the population 16+ for each state.
2.  Controls for Hispanic Origin by age and sex.
3.  Controls by Race, age and sex.
We use all three sets of controls in our weighting program and "rake" through them 6 times so that by the end we come back to all the controls we used. The term estimate refers to population totals derived from CPS by creating "weighted tallies" of any specified socio-economic characteristics of the population. People with similar demographic characteristics should have similar weights. There is one important caveat to remember about this statement. That is that since the CPS sample is actually a collection of 51 state samples, each with its own probability of selection, the statement only applies within state.

### Relevant papers  
Ronny Kohavi and Barry Becker. Data Mining and Visualization, Silicon Graphics.  
e-mail: ronnyk '@' live.com for questions.
Downloaded from openml.org.
``` 

## 3 Metrics
#### 3.1 Fairlearn metric
```python
from fairlearn.metrics import (selection_rate, demographic_parity_difference, demographic_parity_ratio,
                              false_positive_rate, false_negative_rate,
                              false_positive_rate_difference, false_negative_rate_difference,
                               equalized_odds_difference)
# sensitive_features: List, pandas.Series, dict of 1d arrays, numpy.ndarray, pandas.DataFrame
# y can be (List, pandas.Series, numpy.ndarray, pandas.DataFrame)
demographic_parity_ratio(y_true, y_pred, sensitive_features = X_test["sens_attr_name"])
selection_rate(y_true=y_true, y_pred=y_pred)
```


#### 3.2.1 MetricFrame
```python
from fairlearn.metrics import MetricFrame
from fiarlearn.metrics import selection_rate, count

## selection rate: Calculate the fraction of predicted labels matching the ‘good’ outcome.
## pos_label=1 by dafault
metric_frame = MetricFrame(metrics={"selection_rate": selection_rate,
                                    "count": count},
                           sensitive_features = x_test["ensitive_features_name"],
                           y_true=Y_true,
                           y_pred=y_pred)
                           
mf.overall
mf.by_group
```
Note: metrics (callable or dict): <br>
The underlying metric functions which are to be calculated. This can either be a single metric function or a dictionary of functions. These functions must be callable as fn(y_true, y_pred, **sample_params). If there are any other arguments required (such as beta for sklearn.metrics.fbeta_score()) then functools.partial() must be used.

#### 3.2.2 Metric Frame visualization
The simplest way to visualize grouped metrics from the MetricFrame is to take advantage of the inherent plotting capabilities of pandas.DataFrame:
```python
## by group: Return the collection of metrics evaluated for each subgroup in dataframe format

metric_frame.by_group.plot.bar(
    subplots=True,
    layout=[3, 3],
    legend=False,
    figsize=[12, 8],
    title="Show all metrics",
)
```
#### 3.3 Using existing metric definitions from scikit-learn
```python
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)
classifier.fit(X, y_true)
y_pred = classifier.predict(X)
mf = MetricFrame(metrics=accuracy_score, y_true=y_true, y_pred=y_pred, sensitive_features= sens_attr_name)

```

 ## 4 Mitigation
 #### 4.1 Reduction
 In this approach, disparity constraints are cast as Lagrange multipliers, which cause the reweighting and relabelling of the input data. This reduces the problem back to standard machine learning training.
 ```python
 from fairlearn.reductions import DemographicParity
 dp = DemographicParity(difference_bound=number1, ratio_bound=number2, ratio_bound_slack=number3)
 dp.load_data(X, y_true, sensitive_features=sensitive_features)
 dp.gamma(lambda X: y_pred)
 ```
 
