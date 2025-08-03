'''
EXERCISE:
Now you will bring together scaling and model building into a pipeline for cross-validation.
Your task is to build a pipeline to scale features in the music_df dataset and perform grid search cross-validation 
using a logistic regression model with different values for the hyperparameter C. The target variable here is "genre", 
which contains binary values for rock as 1 and any other genre as 0.
StandardScaler, LogisticRegression, and GridSearchCV have all been imported for you.
'''

steps = [('scaler', StandardScaler()),
            ('logreg', LogisticRegression())]

pipeline = Pipeline(steps)
parameters = {"logreg__C": np.linspace(0.001, 1.0, 20)}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

cv = GridSearchCV(pipeline, param_grid=parameters)

cv.fit(X_train, y_train)
print(cv.best_score_, "\n", cv.best_params_)