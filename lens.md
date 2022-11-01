# Python Library Evaluation: Credo Lens

## 1.Version
The newest version was released on Oct 28, 2022, supports python version 3.8+.   <br>
Lens has very frequent updates, and the API strucute and usage changed a lot during these updates. 


## 2.Metric
#### out-of-the-box Matrix
Many metrics are supported out-of-the-box. These metrics can be referenced by string.
```python
from credoai.modules import list_metrics
metrics = list_metrics()
```
<img width="579" alt="Screenshot 2022-10-31 at 9 16 25 PM" src="https://user-images.githubusercontent.com/75053989/199137698-e67fe9ec-4c99-432d-84ae-f09dca114817.png">

## 3. Assessment
Perform specific evaluations on model and/or dataset.
```python
from credoai.evaluators import ModelFairness, Performance
metrics = ['precision_score', 'recall_score', 'equal_opportunity']
ModelFairness(metrics=metrics)
erformance(metrics=metrics
```

## 4. Model
#### 4.1 Classification Model
ClassificationModel serves as an adapter between arbitrary binary or multi-class classification models and the evaluations in Lens.
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
credo_model = ClassificationModel(model_like=model)
```

## Lens
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

