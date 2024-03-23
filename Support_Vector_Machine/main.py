#Linear Support Vector Machine
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)] # petal length, petal width
y = (iris["target"] == 2).astype(np.float64) # Iris virginica

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


#Standardizing the features & Build a linear SVM classifier 

svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge"))

  ])

svm_clf.fit(X, y)

svm_clf.predict([[5.5, 1.7]])

# Reference code for linear SVM classification

#--- Splitting data into 70% training and 30% test data ---

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)

#--- Standardizing the features ---

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

#--- Build a linear SVM classifier ---

from sklearn.svm import LinearSVC
svm_clf = LinearSVC(C=1, loss="hinge")
svm_clf.fit(X_train,y_train)


#--- Test the SVM model ---
y_hat=svm_clf.predict(X_test)


#--- Plot the decision boundary ---

import matplotlib.pyplot as plt
from PlotClassification import plot_decision_regions

import numpy as np
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, classifier=svm_clf, test_idx=range(115, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

