import numpy as np

# XOR truth table
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(a):
    return a * (1 - a)

# Initialize small 2-layer network
np.random.seed(0)
W1 = np.random.randn(2, 2) * 0.5
b1 = np.zeros((1,2))
W2 = np.random.randn(2,1) * 0.5
b2 = np.zeros((1,1))

lr = 0.5
for epoch in range(10000):
    # Forward
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    
    # Error
    error = y - a2
    
    # Back propagation
    delta2 = error * sigmoid_deriv(a2)
    delta1 = np.dot(delta2, W2.T) * sigmoid_deriv(a1)
    
    # Update weights
    W2 += lr * np.dot(a1.T, delta2)
    b2 += lr * np.sum(delta2, axis=0, keepdims=True)
    W1 += lr * np.dot(X.T, delta1)
    b1 += lr * np.sum(delta1, axis=0, keepdims=True)


preds = (a2 > 0.5).astype(int).flatten()
print("\n\n", preds, "\n\n")




