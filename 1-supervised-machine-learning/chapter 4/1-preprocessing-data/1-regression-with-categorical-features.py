'''
EXERCISE:

Now you have created music_dummies, containing binary features for each song's genre, it's time to build a ridge regression model to predict song popularity.
music_dummies has been preloaded for you, along with Ridge, cross_val_score, numpy as np, and a KFold object stored as kf.
The model will be evaluated by calculating the average RMSE, but first, you will need to convert the scores for each fold to positive values and take their square root. This metric shows the average error of our model's predictions, so it can be compared against the standard deviation of the target value—"popularity".
'''
X = music_dummies.drop("popularity", axis=1).values
y = music_dummies["popularity"].values

ridge = Ridge(alpha=0.2)

scores = cross_val_score(ridge, X, y, cv=kf, scoring="neg_mean_squared_error")

rmse = np.sqrt(-scores)
print("Average RMSE: {}".format(np.mean(rmse)))
print("Standard Deviation of the target array: {}".format(np.std(y)))