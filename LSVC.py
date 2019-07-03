# Linear Support Vector Classification
# LinearSVC supports linear kernels and the training time is linear with the number of samples. 

from mnist import MNIST
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

print("Loading dataset...")
mndata = MNIST("./data/")
images, labels = mndata.load_training()

clf = LinearSVC()

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
