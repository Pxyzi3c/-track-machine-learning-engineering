'''
EXCERSICE:
Now you have seen how to evaluate multiple models out of the box, you will build three regression models to predict a song's "energy" levels.
The music_df dataset has had dummy variables for "genre" added. 
Also, feature and target arrays have been created, and these have been split into X_train, X_test, y_train, and y_test.
The following have been imported for you: LinearRegression, Ridge, Lasso, cross_val_score, and KFold.
'''

models = {"Linear Regression": LinearRegression(), "Ridge": Ridge(alpha=0.1), "Lasso": Lasso(alpha=0.1)}
results = []

for model in models.values():
    kf = KFold(n_splits=6, random_state=42, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kf)
    results.append(cv_results)

plt.boxplot(results, labels=models.keys())
plt.show()