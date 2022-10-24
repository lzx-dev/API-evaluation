# Python Library Evaluation: Fairlearn

Fairlearn is an open-source, community-driven project to help data scientists improve fairness of AI systems. The project includes a Python library for fairness assessment and improvement (fairness metrics, mitigation algorithms, plotting, etc.) and educational resources covering organizational and technical processes for unfairness mitigation (comprehensive user guide, detailed case studies, Jupyter notebooks, white papers).

This report will go through the Fairlearn setup, dataset, metric, Mitigation algorithm, reduction algorithm, and functions in other versions and give suggestions from a user's perspective.


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
#### 2.1 Datasets Provided Out Of The Box by Fairlearn
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
#### 2.2 Output Of Data.DESCR
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

#### 2.3 Evaluation
The document of each dataset gives users very detailed context information, like the source of the data and the definition of each attribute. It also clarifies the appropriate tasks(binary classification or regression, how to treat the target variable, which columns caused fairness problems) that should be applied

The output type is flexible.  After users import the dataset and initialize it, the return type can be a bunch object containing data, target, feature names, and description of the data; it can also be simply x and y in Dataframe type or ndarray type.

## 3 Metric
#### 3.1 Fairlearn Metric
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

#### 3.2 MetricFrame
This data structure stores and manipulates disaggregated values for any number of underlying metrics. At least one sensitive feature must be supplied, which is used to split the data into subgroups. 
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

Note: metrics in metricframe(callable or dict): <br>
The underlying metric functions which are to be calculated. This can either be a single metric function or a dictionary of functions. These functions must be callable as fn(y_true, y_pred, **sample_params). If there are any other arguments required (such as beta for sklearn.metrics.fbeta_score()) then functools.partial() must be used.


#### 3.2.1 MetricFrame Visualization
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
#### 3.2.2 Derived Metric
To generate scalar-producing metric functions based on the aggregation methods mentioned above (MetricFrame.group_min(), MetricFrame.group_max(), MetricFrame.difference(), and MetricFrame.ratio()). 
```python
from fairlearn.metrics import make_derived_metric
from fiarlearn.metrics import selection_rate
sr = make_derived_metric(metric=selection_rate, transform='difference')
sr(y_true, y_pred, sensitive_features=A)
```

#### 3.2.3 Control Features
Control features are useful for cases where there is some expected variation with a feature, so we need to compute disparities while controlling for that feature.
```python
cf_metric = MetricFrame(metrics= defined metric,
                             y_true=y_true,
                             y_pred=y_pred,
                             sensitive_features=A,
                         control_features= control_feature)
```


#### 3.3 Using Existing Metric Definitions From scikit-learn
```python
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)
classifier.fit(X, y_true)
y_pred = classifier.predict(X)
mf = MetricFrame(metrics=accuracy_score, y_true=y_true, y_pred=y_pred, sensitive_features= x_test["ensitive_features_name"])
```
```diff
-Note: ValueError: Target is multiclass but average='binary'. Please choose another average setting, one of [None, 'micro', 'macro', 'weighted']. 
will raise when apply multiclass target on sklearn.metrics like recall_score.

```

#### 3.4 Evaluation
MetricFrame is a convient tool to provide users insight on difference between each sensitive groups with inherent plotting capabilities of pandas.DataFrame. It also provides functions to recover the maximum and minimum values of the metric across groups and the difference and ratio between the maximum and minimum.  <br>
However, MetricFrame class did not clarify that metrics like demographic_parity_difference, which measures the difference between groups, can not be added to the frame. Another shortage is that it requires the use of functools.partial() to prebind the required arguments(besides y true and y predict)to the metric function.


 ## 4 Mitigation
 #### 4.1 Pre-Processing
CorrelationRemover applies a linear transformation to the non-sensitive feature columns in order to remove their correlation with the sensitive feature columns while retaining as much information as possible (as measured by the least-squares error).
 ```python
from fairlearn.preprocessing import CorrelationRemover
#Bases: sklearn.base.BaseEstimator, sklearn.base.TransformerMixin

#A component that filters out sensitive correlations in a dataset
cr = CorrelationRemover( sensitive_feature_ids = ["attr_name"])
cf.fit(X)
cf.transform(X)
 ```
 
 #### 4.2 Post-Processing
The predictor’s output is adjusted to fulfill specified parity constraints. The postprocessors learn how to adjust the predictor’s output from the training data.

#### 4.21 ThresholdOptimizer
The classifier is obtained by applying group-specific thresholds to the provided estimator. The thresholds are chosen to optimize the provided performance objective subject to the provided fairness constraints.
 ```python
from fairlearn.postprocessing import ThresholdOptimizer
#The classifier is obtained by applying group-specific thresholds to the provided estimator. The thresholds are chosen to optimize the provided #performance objective subject to the provided fairness constraints
 
 postprocess_est = ThresholdOptimizer(
    estimator=model,  ## A scikit-learn compatible estimator
    constraints="equalized_odds", ## Fairness constraints under which threshold optimization is performed
    prefit=True. ##  If True, avoid refitting the given estimator)
 
 postprocess_est.fit(X_train, Y_train, sensitive_features=A_train)
 postprocess_preds = postprocess_est.predict((X_test, sensitive_features=A_test)
 ```
 
 #### 4.3 Reduction
 In this approach, disparity constraints are cast as Lagrange multipliers, which cause the reweighting and relabelling of the input data. This reduces the problem back to standard machine learning training.
 ```python
 from fairlearn.reductions import DemographicParity
 dp = DemographicParity(difference_bound=number1, ratio_bound=number2, ratio_bound_slack=number3)
 dp.load_data(X, y_true, sensitive_features=sensitive_features)
 ##  Calculate the degree to which constraints are currently violated by the predictor.
 dp.gamma(lambda X: y_pred)
 ```
 
 #### 4.4 Evaluation
Document of each algorithm clarifies the task type(binary classification, regression) and supported fairness definitions(demographic parity, equalized odds, etc.). The type requirement of parameters x and y is flexible: X can be numpy.ndarray, or pandas.DataFrame. y can be numpy.ndarray, pandas.DataFrame, pandas.Series, or list.
 
## 5 Functions in other version
#### 5.1 Dashboard
The Fairlearn dashboard was a Jupyter notebook widget for assessing how a model’s predictions impact different groups (e.g., different ethnicities), and also for comparing multiple models along different fairness and performance metrics.
```python
pip install fairlearn==0.5.0
from fairlearn.widget import FairlearnDashboard
# A_test containts your sensitive features (e.g., age, binary gender)
# sensitive_feature_names contains your sensitive feature names
# y_true contains ground truth labels
# y_pred contains prediction labels

FairlearnDashboard(sensitive_features=A_test,
                   sensitive_feature_names=['BinaryGender', 'Age'],
                   y_true=Y_test.tolist(),
                   y_pred=[y_pred.tolist()])
```

```diff
-Note: The FairlearnDashboard will move from Fairlearn to the raiwidgets package after the v0.5.0 release.

```

#### 5.2 Evaluation
The Fairlearn dashboard provides a visual way to compare metrics between models as well as compare metrics for different groups on a single model. However, it can only be displayed on Jupyter notebook. 
