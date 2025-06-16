import numpy as np
import os

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x): return x * (1 - x)

def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(float)

class MLP:
    def __init__(self, input_size, hidden_size1=10, hidden_size2=6, activation='relu', learning_rate=0.01):
        self.lr = learning_rate
        self.activation_name = activation

        # Pesos para capa 1
        self.w1 = np.random.randn(input_size, hidden_size1)
        self.b1 = np.zeros((1, hidden_size1))

        # Pesos para capa 2
        self.w2 = np.random.randn(hidden_size1, hidden_size2)
        self.b2 = np.zeros((1, hidden_size2))

        # Capa de salida
        self.w3 = np.random.randn(hidden_size2, 1)
        self.b3 = np.zeros((1, 1))

        # Funciones de activación
        self.act = sigmoid if activation == 'sigmoid' else relu
        self.act_deriv = sigmoid_deriv if activation == 'sigmoid' else relu_deriv

    def forward(self, X):
        self.z1 = X @ self.w1 + self.b1
        self.a1 = self.act(self.z1)

        self.z2 = self.a1 @ self.w2 + self.b2
        self.a2 = self.act(self.z2)

        self.z3 = self.a2 @ self.w3 + self.b3
        self.a3 = sigmoid(self.z3)  # salida final binaria
        return self.a3

    def backward(self, X, y):
        m = y.shape[0]

        # Salida -> capa 2
        dz3 = self.a3 - y
        dw3 = (self.a2.T @ dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m

        # capa 2 -> capa 1
        dz2 = dz3 @ self.w3.T * self.act_deriv(self.a2)
        dw2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # capa 1 -> entrada
        dz1 = dz2 @ self.w2.T * self.act_deriv(self.a1)
        dw1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Actualizar pesos
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        self.w3 -= self.lr * dw3
        self.b3 -= self.lr * db3

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            y_pred = self.forward(X)

            # Cross-entropy loss
            loss = -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))
            self.backward(X, y)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss {loss:.4f}")

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

    def guardar_modelo(self, archivo):
        # Crear carpeta 'models' si no existe
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        ruta = os.path.join(models_dir, archivo + ".npz")
        np.savez(ruta,
                 w1=self.w1, b1=self.b1,
                 w2=self.w2, b2=self.b2,
                 w3=self.w3, b3=self.b3)
        print(f"✅ Modelo guardado en '{ruta}'")

    def cargar_modelo(self, archivo):
        models_dir = "models"
        ruta = os.path.join(models_dir, archivo if archivo.endswith('.npz') else archivo + ".npz")
        datos = np.load(ruta)
        self.w1 = datos['w1']
        self.b1 = datos['b1']
        self.w2 = datos['w2']
        self.b2 = datos['b2']
        self.w3 = datos['w3']
        self.b3 = datos['b3']
        print(f"📥 Modelo cargado desde '{ruta}'")
