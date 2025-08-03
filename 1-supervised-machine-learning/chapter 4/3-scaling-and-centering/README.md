# Centering and scaling

## Why scale our data?

* Many models use some form of distance to inform them
* Features on larger scales can disproportionately influence the model
* **Example:** KNN uses distance explicitly when making predictions
* We want features to be on  a similar scale
* Normalizing or standardizing (scaling and centering)

## How to scale our data

* Subtract the mean and divide by variance
	* All features are centered around zero and have a variance of one
	* This is called **standardization**
* Can also subtract the minimum and divide by the range
	* Minimum zero and maximum one
* Can also __normalize__ so the data ranges from -1 to +1
* See scikit-learn docs for further details

## Scaling in scikit-learn

```python
from sklearn.preprocessing import StandardScaler

X = music_df.drop("genre", axis=1).values
y = music_df["genre"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_trasform(X_train)
X_test_scaled = scaler.transform(X_test)
print(np.mean(X), np.std(X))
print(np.mean(X_train_scaled), np.std(X_train_scaled))
```
```bash
# Output:
19801.425, 71343.529
2.260e-17, 1.0
```

## Scaling in a pipeline

```python
steps = [('scaler', StandardScaler()),
	      ('knn', KNeighborsClassifier(n_neighbors=6))]
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

knn_scaled = pipeline.fit(X_train, y_train)
y_pred = knn.scaled.predict(X_test)
print(knn_scaled.score(X_test, y_test))
```
```bash
# Output:
0.81
```

## Comparing performance using unscaled data

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

knn_unscaled = KNeighborsClassifier(n_neighbors=6).fit(X_train, y_train)
print(knn_unscaled.score(X_test, y_test))
```
```bash
**Output:**
0.53
```

## CV and scaling in a pipeline

```python
from sklearn.model_selection import GridSearchCV

steps = [('scaler', StandardScaler()),
	      ('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)
parameters = {"knn__n_neighbors": np.arange(1, 50)}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

cv = GridSearchCV(pipeline, param_grid=parameters)
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
```

## Checking model parameters
```python
print(cv.best_score_)
```
```bash
# Output:
0.819999
```
```python
print(cv.best_params_)
```
```bash
# Output:
{'knn__n_neighbors': 12}
```