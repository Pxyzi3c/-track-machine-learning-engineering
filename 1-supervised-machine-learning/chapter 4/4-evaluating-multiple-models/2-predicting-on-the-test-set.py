'''
EXERCISE:
In the last exercise, linear regression and ridge appeared to produce similar results. 
It would be appropriate to select either of those models; however, you can check predictive performance on the test set to see if either one can outperform the other.
You will use root mean squared error (RMSE) as the metric. The dictionary models, containing the names and instances of the two models, 
has been preloaded for you along with the training and target arrays X_train_scaled, X_test_scaled, y_train, and y_test.
'''
from sklearn.metrics import root_mean_squared_error

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    test_rmse = root_mean_squared_error(y_test, y_pred)
    print("{} Test Set RMSE: {}".format(name, test_rmse))