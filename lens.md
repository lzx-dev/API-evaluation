# Python Library Evaluation: Credo Lens
An open source Responsible AI Assessment Framework for Data Scientists to rapidly assess models and generate results for governance review within the Credo AI app.

## 1.Version 1.1.0
The newest version was released on Oct 28, 2022, supports python version 3.8+.   <br>
Lens has very frequent updates, and the API strucute and usage changed a lot during these updates. 


## 2.Metric
#### 2.1 Out-of-the-box Metrics
Many metrics are supported out-of-the-box. These metrics can be referenced by string.
```python
from credoai.modules import list_metrics
metrics = list_metrics()
```
<img width="579" alt="Screenshot 2022-10-31 at 9 16 25 PM" src="https://user-images.githubusercontent.com/75053989/199137698-e67fe9ec-4c99-432d-84ae-f09dca114817.png">

#### 2.2 Custom Metrics
```python
from credoai.modules import Metric
Metric(name = 'metric',
       metric_category = "binary_classification",
       fun = fun)
#fun (callable, optional) – The function definition of the metric. 
#If none, the metric cannot be used and is only defined for documentation purposes
```

## 3.Evaluator
Evaluators are the classes that perform specific functions on a model and/or data. These can include assessing the model for fairness, or profiling a data. Evaluators are constantly being added to the framework, which creates Lens’s standard library.
```python
from credoai.evaluators import ModelFairness, Performance
metrics = ['precision_score', 'recall_score', 'equal_opportunity']
ModelFairness(metrics=metrics)
Performance(metrics=metrics
```

## 4.Model&Data
#### 4.1 Classification Model
ClassificationModel serves as an adapter between arbitrary binary or multi-class classification models and the evaluations in Lens.
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
credo_model = ClassificationModel(model_like=model)
```

#### 4.2 Tabular Data
Data type artifact, like TabularData serve as adapters between datasets and the evaluators in Lens. 
When you pass data to a Data artifact, the artifact performs various steps of validation, and formats them so that they can be used by evaluators. The aim of this procedure is to preempt errors down the line.
```python
credo_data = TabularData(
    name="UCI-credit-default",
    X=X_test,
    y=y_test,
    sensitive_features=sensitive_features_test,
)
```

## 5.Lens
Lens is the principle interface to the Credo AI Lens assessment framework.
```python
credo_model = ClassificationModel(name="credit_default_classifier", model_like=model)
credo_data = TabularData(
    name="UCI-credit-default",
    X=X_test,
    y=y_test,
    sensitive_features=sensitive_features_test,
)
lens = Lens(model=credo_model, assessment_data=credo_data)
metrics = ['precision_score', 'recall_score', 'equal_opportunity']
lens.add(ModelFairness(metrics=metrics))
results = lens.get_results()
```
<img width="443" alt="Screenshot 2022-11-01 at 9 46 56 AM" src="https://user-images.githubusercontent.com/75053989/199248370-7068387b-27ca-4917-89c3-a5c77cd609b0.png">


## 6.Credo AI Platform
Connecting Lens to the Governance App requires that you have already defined a Use-Case and defined a policy pack defining how associated models and data should be assessed.
```python
from credoai.governance import Governance
gov = Governance()
url = 'your assessment url'
gov.register(assessment_plan_url=url)
lens.send_to_governance()
#Export to platform
gov.export()
```

## 7.Evaluation
When installing Lens with pip, it does not guarantee to provide the latest version. For example, the newest version is 1.1.0, but the version from the pip install only gives me 0.2.1 unless I specify the version I want. The structure and usage of the API changes a lot during updates and only the latest verion document is provided, which is not user-friendly since users with previos versions would encounter many errors if they use the current document as reference.

Another problem is as a machine learning fairness-related API, it does not provide any mitigation algorithms and no visualization functions unless connected to the Credo Application platform

