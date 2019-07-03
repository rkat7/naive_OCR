# Works by using a multitude of decision trees and it selects the class that is the most often predicted by the trees.
# A decision tree contains at each vertex a "question" and each descending edge is an "answer" to that question. The leaves of the tree are the possible outcomes.
# A decision tree can be built automatically from a training set.
# Each tree of the forest is created using a random sample of the original training set, and by considering only a subset of the features (typically the square root of the number of features).
# The number of trees is controlled by cross-validation.

from mnist import MNIST
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Loading dataset...")
mndata = MNIST("./data/")
images, labels = mndata.load_training()

clf = RandomForestClassifier(n_estimators=100)

# Train on the first 10000 images:
train_x = images[:10000]
train_y = labels[:10000]

print("Train model")
clf.fit(train_x, train_y)

# Test on the next 1000 images:
test_x = images[10000:11000]
expected = labels[10000:11000].tolist()

print("Compute predictions")
predicted = clf.predict(test_x)

print("Accuracy: ", accuracy_score(expected, predicted))
