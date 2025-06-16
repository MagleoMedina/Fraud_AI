import pandas as pd
from datos import generar_dataset_csv
from mlp import MLP
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# 1. Generar dataset (solo si no existe para no sobreescribir)
if not os.path.exists("dataset.csv"):
    generar_dataset_csv()

# 2. Cargar dataset
df = pd.read_csv("dataset.csv")

# 3. Preparar datos
X = df.drop("isFraud", axis=1).values
y = df["isFraud"].values.reshape(-1, 1)

# 4. Normalizar características
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 5. Dividir en entrenamiento y prueba (para entrenamiento solo usamos X_train, y_train)
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.4, random_state=42)

# 6. Crear y entrenar modelo
mlp = MLP(input_size=6, hidden_size1=10, hidden_size2=6, activation='relu', learning_rate=0.01)
mlp.train(X_train, y_train, epochs=1000)

# 7. Guardar modelo entrenado
mlp.guardar_modelo("modelo_mlp")

# 8. Guardar scaler para normalizar datos en prueba (opcional)
import joblib
joblib.dump(scaler, "scaler.pkl")

print("✅ Entrenamiento completado y modelo guardado.")
