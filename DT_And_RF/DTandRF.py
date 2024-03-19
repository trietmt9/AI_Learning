#======= Decision tree for iris classification =======

#--- Import iris dataset ---

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target


#--- Splitting data into 70% training and 30% test data ---

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)


#--- Build a decicion tree ---

from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X_train, y_train)


#--- Plot the trained tree ---

from sklearn import tree
tree.plot_tree(tree_clf)


# #--- Google Drive Path Setting ---

# import os
# from google.colab import drive
# drive.mount('/content/drive')
# os.chdir('/content/drive/My Drive/Colab Notebooks/AIPR112') 


#--- Plot the decision boundary ---

import matplotlib.pyplot as plt
from PlotClassification import plot_decision_regions

import numpy as np
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, classifier=tree_clf, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()




#--- Model evaluation ---

from sklearn.metrics import accuracy_score
y_test_pred=tree_clf.predict(X_test)
accuracy_score(y_test, y_test_pred)


#--- Confusion matrix ----

from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(y_test, y_test_pred)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()





#--- Precision, recall, F1-score ---

from sklearn.metrics import precision_score, recall_score
precision_score(y_test, y_test_pred, average = 'macro')

recall_score(y_test, y_test_pred, average = 'macro')

from sklearn.metrics import f1_score
f1_score(y_test, y_test_pred, average = 'macro')


#==================================================================================

#=== Decision tree for iris classification ===

from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)

rnd_clf.fit(X_train, y_train)

y_pred = rnd_clf.predict(X_test)