# mlp.py
import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x): return x * (1 - x)

def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(float)

class MLP:
    def __init__(self, input_size, hidden_size, activation='sigmoid', learning_rate=0.01):
        self.lr = learning_rate
        self.activation_name = activation

        self.w1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, 1)
        self.b2 = np.zeros((1, 1))

        self.act = sigmoid if activation == 'sigmoid' else relu
        self.act_deriv = sigmoid_deriv if activation == 'sigmoid' else relu_deriv

    def forward(self, X):
        self.z1 = X @ self.w1 + self.b1
        self.a1 = self.act(self.z1)
        self.z2 = self.a1 @ self.w2 + self.b2
        self.a2 = sigmoid(self.z2)  # salida binaria
        return self.a2

    def backward(self, X, y):
        m = y.shape[0]
        dz2 = self.a2 - y
        dw2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = dz2 @ self.w2.T * self.act_deriv(self.a1)
        dw1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = np.mean((y_pred - y)**2)
            self.backward(X, y)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss {loss:.4f}")

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)
    
    def guardar_modelo(self, archivo):
        np.savez(archivo,
                 w1=self.w1,
                 b1=self.b1,
                 w2=self.w2,
                 b2=self.b2)
        print(f"✅ Modelo guardado en '{archivo}.npz'")

    def cargar_modelo(self, archivo):
        datos = np.load(archivo)
        self.w1 = datos['w1']
        self.b1 = datos['b1']
        self.w2 = datos['w2']
        self.b2 = datos['b2']
        print(f"📥 Modelo cargado desde '{archivo}'")
