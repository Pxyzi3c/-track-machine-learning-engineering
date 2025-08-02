'''
GridSearchCV in scikit-learn:

from sklearn.model_selection import GridSearchCV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {"alpha": np.arange(0.0001, 1, 10),
			"solver": ["sag", "lsqr"]}
ridge = Ridge()
ridge_cv = GridSearcchCV(ridge, param_grid, cv=kf)
ridge_cv.fit(X_train, y_train)
print(ridge_cv.best_params_, ridge_cv.best_score_)
'''

'''
EXERCISE:
Now you have seen how to perform grid search hyperparameter tuning, you are going to build a lasso regression model with optimal hyperparameters to predict blood glucose levels using the features in the diabetes_df dataset.
X_train, X_test, y_train, and y_test have been preloaded for you. A KFold() object has been created and stored for you as kf, along with a lasso regression model as lasso.
'''
from sklearn.model_selection import GridSearchCV

param_grid = {"alpha": np.linspace(0.00001, 1, 20)}

lasso_cv = GridSearchCV(lasso, param_grid, cv=kf)

lasso_cv.fit(X_train, y_train)
print("Tuned lasso paramaters: {}".format(lasso_cv.best_params_))
print("Tuned lasso score: {}".format(lasso_cv.best_score_))