# Evaluating Multiple Models

## Different models for different problems

**Some guiding principles**
* Size of the dataset
	* Fewer features = simpler model, faster training time
	* Some models require large amounts of data to perform well
* Interpretability
	* Some models are easier to explain, which can be important for stakeholders
	* Linear regression has high interpretability, as we can understand the coefficients
* Flexibility
	* May improve accuracy, by making fewer assumptions about data
	* KNN is a more flexible model, doesn't assume any linear relationships

## It's all in the metrics

* Regression model performance:
	* RMSE
	* R-squared
* Classification model performance:
	* Accuracy
	* Confusion matrix
	* Precision, recall, F1-score
	* ROC AUC
* Train several models and evaluate performance out of the box.

## A note on scaling

* Models affected by scaling:
	* KNN
	* Linear Regression (plus Ridge, Lasso)
	* Logistic Regression
	* Artificial Neural Network
* Best to scale our data before evaluating models

## Evaluating classification models

```python
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

X = music.drop("genre", axis=1).values
y = music["genre"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_trasform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {"Logistic Regression": LogisticRegression(), "KNN": KNeighborsClassifier(), "Decision Tree": DecisionTreeClassifier()}
results = []
for model in models.values():
	kf = KFold(n_splits=6, random_state=42, shuffle=True)
	cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kf)
	results.append(cv_results)
plt.boxplot(results, labels=models.keys())
plt.show()
```

## Test set performance

```python
for name, model in models.items():
	model.fit(X_train_scaled, y_train)
	test_score = model.score(X_test_scaled, y_test)
	print("{} Test Set Accuracy: {}.format(name, test_score))
```
```bash
# Ouput:
Logistic Regression Test Set Accuracy: 0.844
KNN Test Set Accuracy: 0.82
Decision Tree Test Set Accuracy: 0.832
```