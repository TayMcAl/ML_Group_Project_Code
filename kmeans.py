import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Load the data
def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data('../input/mnist-numpy/mnist.npz')

# Reshape data from shape 28 x 28 to 784
X = x_train.reshape(len(x_train),-1)
X_test = x_test.reshape(len(x_test),-1)


print('Training Data: {}'.format(x_train.shape))
print('Training Labels: {}'.format(y_train.shape))

print('Testing Data: {}'.format(x_test.shape))
print('Testing Labels: {}'.format(y_test.shape))

# Normalize the data to 0 - 1
X = X.astype(float) / 255.
X_test = X_test.astype(float) / 255.

class K_Means:
    def __init__(self, k=10, tol=0.001, max_iter=300000):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):

        self.centroids = {}

        for i in range(self.k):
            rand = np.random.randint(0,len(data))
            self.centroids[i] = data[rand]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if ((original_centroid.any() != 0) and (np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol)):
                    optimized = False

            if optimized:
                break

    def assess(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        return distances
    
    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

def accuracy(model, labels):
    predicted_labels = []
    for data in X:
        predicted_labels.append(model.predict(data))
    return np.mean(predicted_labels == labels)

def matrix(model, labels):
    predicted_labels = []
    for data in X:
        predicted_labels.append(model.predict(data))
        
    return confusion_matrix(labels, predicted_labels)

trains = []
tests = []
models = []
clusters = [10]
for cluster in clusters:
    r = 5
    k = cluster
    for i in range(r):
        model = K_Means(k)
        model.fit(X)
        models.append(model)
        trains.append(accuracy(model, y_train))
        tests.append(accuracy(model, y_test))
best_train = max(trains) 
best_test = max(tests)
print("Number of clusters: ", k)
print("Best accuracy: ", best_train)
index = trains.index(best_train)
print("From the Model #", index)
print(matrix(models[trains.index(best_train)], y_train))



