import pandas as pd
from mlp import MLP
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.model_selection import train_test_split


# 1. Cargar dataset para prueba
df = pd.read_csv("dataset.csv")

X = df.drop("isFraud", axis=1).values
y = df["isFraud"].values.reshape(-1, 1)

# 2. Cargar scaler y normalizar datos (solo test)
if os.path.exists("scaler.pkl"):
    scaler = joblib.load("scaler.pkl")
    X = scaler.transform(X)
else:
    print("⚠️ No se encontró el scaler.pkl, normalizando con StandardScaler nuevo.")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

# 3. Dividir datos para prueba
_, X_test, _, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# 4. Cargar modelo
modelo_path = "modelo_mlp.npz"
if not os.path.exists(modelo_path):
    raise FileNotFoundError(f"No se encontró el archivo de modelo '{modelo_path}'. Entrena primero el modelo.")

mlp = MLP(input_size=6, hidden_size1=10, hidden_size2=6, activation='relu', learning_rate=0.01)
mlp.cargar_modelo(modelo_path)

# 5. Realizar predicciones
y_pred = mlp.predict(X_test)

# 6. Evaluar resultados
print("\n✅ Reporte de Clasificación:")
print(classification_report(y_test, y_pred))
print("✅ Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print(f"✅ Exactitud del modelo sobre datos no vistos: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 7. Gráfico combinado
reales_legitimas = int((y_test == 0).sum())
reales_fraude = int((y_test == 1).sum())
pred_legitimas = int((y_pred == 0).sum())
pred_fraude = int((y_pred == 1).sum())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(['Reales', 'Predichas'], [reales_legitimas, pred_legitimas], color=['skyblue', 'orange'])
axes[0].set_title('Transacciones Legítimas')
axes[0].set_ylabel('Cantidad')

axes[1].bar(['Reales', 'Predichas'], [reales_fraude, pred_fraude], color=['skyblue', 'orange'])
axes[1].set_title('Transacciones Fraudulentas')

plt.suptitle('Comparación de Transacciones Reales vs Predichas')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("grafico_comparativo_transacciones.png")

print("📊 Gráfico combinado guardado como 'grafico_comparativo_transacciones.png'")
