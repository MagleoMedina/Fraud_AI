import pandas as pd
from generador_dataset import generar_dataset_csv
from mlp import MLP
from sklearn.preprocessing import StandardScaler
import os
import joblib

# 1. Generar dataset (solo si no existe para no sobreescribir)
if not os.path.exists("dataset/dataset60.csv") or not os.path.exists("dataset/dataset40.csv"):
    print("⚠️ Archivos 'dataset60.csv' o 'dataset40.csv' no encontrados. Generando nuevamente...")
    generar_dataset_csv()

# 2. Cargar dataset de entrenamiento (60%)
df = pd.read_csv("dataset/dataset60.csv")

# 3. Preparar datos
X = df.drop("isFraud", axis=1).values
y = df["isFraud"].values.reshape(-1, 1)

# 4. Normalizar características
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Usar el 100% de los datos de dataset60.csv para entrenamiento
X_train = X
y_train = y

# 6. Crear y entrenar modelo
mlp = MLP(input_size=6, hidden_size1=10, hidden_size2=6, activation='relu', learning_rate=0.01)
mlp.train(X_train, y_train, epochs=1500)

# Guardar modelo entrenado y métricas
mlp.guardar_modelo("modelo_mlp")
mlp.guardar_metricas("metricas_entrenamiento")

# 8. Guardar scaler para normalizar datos en prueba 
models_dir = "models"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    
joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

print("Entrenamiento completado y modelo guardado.")
