from sklearn.neural_network import MLPClassifier
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

clf = MLPClassifier(hidden_layer_sizes=(2,), activation='tanh', max_iter=5000, random_state=42)
clf.fit(X, y)
print("\n\n", clf.predict(X), "\n\n")

