'''
EXERCISE:
The ROC curve you plotted in the last exercise looked promising.

Now you will compute the area under the ROC curve, along with the other classification metrics you have used previously.

The confusion_matrix and classification_report functions have been preloaded for you, along with the logreg model you previously built, plus X_train, X_test, y_train, y_test. Also, the model's predicted test set labels are stored as y_pred, and probabilities of test set observations belonging to the positive class stored as y_pred_probs.

A knn model has also been created and the performance metrics printed in the console, so you can compare the roc_auc_score, confusion_matrix, and classification_report between the two models.
'''

from sklearn.metrics import roc_auc_score

print(roc_auc_score(y_test, y_pred_probs))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))