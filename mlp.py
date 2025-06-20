import numpy as np
import os

# Función de activación sigmoide
def sigmoid(x): return 1 / (1 + np.exp(-x))
# Derivada de la función sigmoide
def sigmoid_deriv(x): return x * (1 - x)

# Función de activación ReLU
def relu(x): return np.maximum(0, x)
# Derivada de la función ReLU
def relu_deriv(x): return (x > 0).astype(float)

class MLP:
    def __init__(self, input_size, hidden_size1=10, hidden_size2=6, activation='relu', learning_rate=0.01):
        """
        Inicializa la red neuronal multicapa (MLP) con los tamaños de las capas y la función de activación.
        """
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
        """
        Realiza la propagación hacia adelante (forward pass) de la red.
        Devuelve la salida predicha.
        """
        self.z1 = X @ self.w1 + self.b1
        self.a1 = self.act(self.z1)

        self.z2 = self.a1 @ self.w2 + self.b2
        self.a2 = self.act(self.z2)

        self.z3 = self.a2 @ self.w3 + self.b3
        self.a3 = sigmoid(self.z3)  # salida final binaria
        return self.a3

    def backward(self, X, y):
        """
        Realiza la retropropagación (backpropagation) para actualizar los pesos de la red.
        """
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

    def train(self, X, y, epochs=1500):
        """
        Entrena la red neuronal durante un número de épocas usando los datos X e y.
        Guarda el historial de pérdida y métricas.
        """
        self.loss_history = []  # Store loss for each epoch
        self.metrics_history = []  # Store metrics for each epoch

        for epoch in range(epochs):
            y_pred = self.forward(X)

            # Cross-entropy loss
            loss = -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))
            self.loss_history.append(loss)

            # Calculate metrics (accuracy for simplicity)
            accuracy = np.mean((y_pred > 0.5).astype(int) == y)
            self.metrics_history.append({'accuracy': accuracy})

            self.backward(X, y)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss {loss:.4f}, Accuracy {accuracy:.4f}")

    def predict(self, X):
        """
        Realiza una predicción binaria (0 o 1) sobre los datos X.
        """
        return (self.forward(X) > 0.65).astype(int)
    
    def predict_proba(self, X):
        """
        Devuelve la probabilidad predicha (salida de la capa sigmoide) para los datos X.
        """
        return self.forward(X)

    def guardar_modelo(self, archivo):
        """
        Guarda los pesos del modelo en un archivo dentro de la carpeta 'models'.
        """
        # Crear carpeta 'models' si no existe
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        ruta = os.path.join(models_dir, archivo + ".npz")
        np.savez(ruta,
                 w1=self.w1, b1=self.b1,
                 w2=self.w2, b2=self.b2,
                 w3=self.w3, b3=self.b3)
        print(f"Modelo guardado en '{ruta}'")

    def cargar_modelo(self, archivo):
        """
        Carga los pesos del modelo desde un archivo en la carpeta 'models'.
        """
        models_dir = "models"
        # Check if 'models' is already in the path
        ruta = archivo if os.path.dirname(archivo) == models_dir else os.path.join(models_dir, archivo if archivo.endswith('.npz') else archivo + ".npz")
        datos = np.load(ruta)
        self.w1 = datos['w1']
        self.b1 = datos['b1']
        self.w2 = datos['w2']
        self.b2 = datos['b2']
        self.w3 = datos['w3']
        self.b3 = datos['b3']
        print(f"📥 Modelo cargado desde '{ruta}'")

    def guardar_scaler(self, scaler, archivo):
        """
        Guarda el objeto scaler (por ejemplo, StandardScaler) en la carpeta 'models'.
        """
        import joblib
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        ruta = os.path.join(models_dir, archivo if archivo.endswith('.pkl') else archivo + ".pkl")
        joblib.dump(scaler, ruta)
        print(f"✅ Scaler guardado en '{ruta}'")

    def cargar_scaler(self, archivo):
        """
        Carga el objeto scaler desde la carpeta 'models'.
        """
        import joblib
        models_dir = "models"
        # Check if 'models' is already in the path
        ruta = archivo if os.path.dirname(archivo) == models_dir else os.path.join(models_dir, archivo if archivo.endswith('.pkl') else archivo + ".pkl")
        scaler = joblib.load(ruta)
        print(f"📥 Scaler cargado desde '{ruta}'")
        return scaler

    def guardar_metricas(self, archivo):
        """
        Guarda el historial de pérdida y métricas en la carpeta 'metrics'.
        """
        metrics_dir = "metrics"
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)
        ruta = os.path.join(metrics_dir, archivo if archivo.endswith('.npz') else archivo + ".npz")
        np.savez(ruta, loss_history=self.loss_history, metrics_history=self.metrics_history)
        print(f"✅ Métricas guardadas en '{ruta}'")

    def cargar_metricas(self, archivo):
        """
        Carga el historial de pérdida y métricas desde la carpeta 'metrics'.
        """
        metrics_dir = "metrics"
        ruta = archivo if os.path.dirname(archivo) == metrics_dir else os.path.join(metrics_dir, archivo if archivo.endswith('.npz') else archivo + ".npz")
        datos = np.load(ruta, allow_pickle=True)
        self.loss_history = datos['loss_history']
        self.metrics_history = datos['metrics_history']
        print(f"📥 Métricas cargadas desde '{ruta}'")
