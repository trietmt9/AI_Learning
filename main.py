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

