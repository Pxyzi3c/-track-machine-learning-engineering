'''
EXERCISE:
Now you have seen the benefits of scaling your data, you will use a pipeline to preprocess the music_df features 
and build a lasso regression model to predict a song's loudness.
X_train, X_test, y_train, and y_test have been created from the music_df dataset, where the target is "loudness"
and the features are all other columns in the dataset. Lasso and Pipeline have also been imported for you.
Note that "genre" has been converted to a binary feature where 1 indicates a rock song, and 0 represents other genres.
'''

from sklearn.preprocessing import StandardScaler

steps = [('scaler', StandardScaler()),
          ('lasso', Lasso(alpha=0.5))]

pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)

print(pipeline.score(X_test, y_test))