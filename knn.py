# Represent the training data as a set of points in the feature space (e.g. for objects with 2 features, they are points of the 2D plan).
# Classification is based on the K closest points of the training set to the object we wish to classify. The object is classified by a majority vote of its k-nearest neighbors.
#The training phase of the K-Nearest Neighbors algorithm is fast, however the classification phase may be slow due to computation of K distances.

from mnist import MNIST
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

print("Loading dataset...")
mndata = MNIST("./data/")
images, labels = mndata.load_training()

clf = KNeighborsClassifier()

# Train on the first 10000 images:
train_x = images[:10000]
train_y = labels[:10000]

print("Train model")
clf.fit(train_x, train_y)

# Test on the next 100 images:
test_x = images[10000:10100]
expected = labels[10000:10100].tolist()

print("Compute predictions")
predicted = clf.predict(test_x)

print("Accuracy: ", accuracy_score(expected, predicted))
