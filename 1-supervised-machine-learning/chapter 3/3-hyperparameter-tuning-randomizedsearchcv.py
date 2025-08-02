'''
RandomizedSearchCV in scikit-learn:

from sklearn.model_selection import RandomizedSearchCV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {'alpha': np.arange(0.0001, 1, 10),
			"solver": ['sag', 'lsqr']}
ridge = Ridge()
ridge_cv = RandomizedSearchCV(ridge, param_grid, cv=kf, n_iter=2)
ridge_cv.fit(X_train, y_train)
print(ridge_cv.best_params_, ridge_cv.best_score_)
=====================================================
Evaluating on the test set:

test_score = ridge_cv.score(Y_test, y_test)
print(test_score)
'''

'''EXERCISE:
As you saw, GridSearchCV can be computationally expensive, especially if you are searching over a large hyperparameter space. In this case, you can use RandomizedSearchCV, which tests a fixed number of hyperparameter settings from specified probability distributions.
Training and test sets from diabetes_df have been pre-loaded for you as X_train. X_test, y_train, and y_test, where the target is "diabetes". A logistic regression model has been created and stored as logreg, as well as a KFold variable stored as kf.
You will define a range of hyperparameters and use RandomizedSearchCV, which has been imported from sklearn.model_selection, to look for optimal hyperparameters from these options.
'''
params = {'penalty': ["l1", "l2"],
        'tol': np.linspace(0.0001, 1.0, 50),
        'C': np.linspace(0.1, 1.0, 50),
        'class_weight': ["balanced", {0:0.8, 1:0.2}]}
logreg_cv = RandomizedSearchCV(logreg, params, cv=kf)

logreg_cv.fit(X_train, y_train)

print("Tuned Logistic Regession Parameters: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Score: {}".format(logreg_cv.best_score_))