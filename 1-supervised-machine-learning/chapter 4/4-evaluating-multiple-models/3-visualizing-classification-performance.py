'''
EXERCISE:
In this exercise, you will be solving a classification problem where the "popularity" column in the music_df dataset has 
been converted to binary values, with 1 representing popularity more than or equal to the median for the "popularity" column, and 0 indicating popularity below the median.
Your task is to build and visualize the results of three different models to classify whether a song is popular or not.
The data has been split, scaled, and preloaded for you as X_train_scaled, X_test_scaled, y_train, and y_test. 
Additionally, KNeighborsClassifier, DecisionTreeClassifier, and LogisticRegression have been imported.
'''

models = {"Logistic Regression": LogisticRegression(), "KNN": KNeighborsClassifier(), "Decision Tree Classifier": DecisionTreeClassifier()}

results = []
for model in models.values():
    kf = KFold(n_splits=6, random_state=12, shuffle=True)
    cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kf)
    results.append(cv_results)

plt.boxplot(results, labels=models.keys())
plt.show()