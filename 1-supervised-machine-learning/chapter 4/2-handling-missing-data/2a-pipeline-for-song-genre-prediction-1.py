'''
EXERCISE:
Now it's time to build a pipeline. It will contain steps to impute missing values using the mean for each feature and build a KNN model for the classification of song genre.
The modified music_df dataset that you created in the previous exercise has been preloaded for you, along with KNeighborsClassifier and train_test_split.
'''

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

imputer = SimpleImputer()
knn = KNeighborsClassifier(n_neighbors=3)

steps = [("imputer", imputer),
         ("knn", knn)]