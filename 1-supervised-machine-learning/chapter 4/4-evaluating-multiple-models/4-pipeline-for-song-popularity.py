'''
EXERCISE:
For the final exercise, you will build a pipeline to impute missing values, scale features, and perform hyperparameter tuning of a logistic regression model. 
The aim is to find the best parameters and accuracy when predicting song genre!
All the models and objects required to build the pipeline have been preloaded for you.
'''

steps = [('imp_mean', SimpleImputer()),
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression())]

pipeline = Pipeline(steps)
params = {"logreg__solver": ["newton-cg", "saga", "lbfgs"],
         "logreg__C": np.linspace(0.001, 1.0, 10)}

tuning = GridSearchCV(pipeline, param_grid=params)
tuning.fit(X_train, y_train)
y_pred = tuning.predict(X_test)

print("Tuned Logistic Regression Parameters: {}, Accuracy: {}".format(tuning.best_params_, tuning.score(X_test, y_test)))