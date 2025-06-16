import pandas as pd
from mlp import MLP
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

# Crear la carpeta 'graphics' si no existe
output_dir = "graphics"
os.makedirs(output_dir, exist_ok=True)

# 1. Cargar dataset para prueba (40%)
df = pd.read_csv("dataset/dataset40.csv")
X = df.drop("isFraud", axis=1).values
y = df["isFraud"].values.reshape(-1, 1)

# 2. Cargar scaler y normalizar datos
scaler_path = os.path.join("models", "scaler.pkl")
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    X = scaler.transform(X)
else:
    print("⚠️ No se encontró el scaler.pkl, normalizando con StandardScaler nuevo.")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

# 3. Dividir datos para prueba (ya no es necesario dividir, usar todo X y y)
X_test = X
y_test = y

# 4. Cargar modelo
modelo_path = os.path.join("models", "modelo_mlp.npz")
if not os.path.exists(modelo_path):
    raise FileNotFoundError(f"No se encontró el archivo de modelo '{modelo_path}'. Entrena primero el modelo.")

mlp = MLP(input_size=6, hidden_size1=10, hidden_size2=6, activation='relu', learning_rate=0.01)
mlp.cargar_modelo(modelo_path)

# 5. Realizar predicciones
#y_pred = mlp.predict(X_test)

y_prob = mlp.predict_proba(X_test)
y_pred = (y_prob >= 0.65).astype(int)  # Puedes probar con 0.6, 0.65, 0.7...


# 6. Evaluar resultados
print("\n✅ Reporte de Clasificación:")
print(classification_report(y_test, y_pred))
print("✅ Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print(f"✅ Exactitud del modelo sobre datos no vistos: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 7. Estadísticas por clase
fraudes_reales = int((y_test == 1).sum())
fraudes_detectados = int(((y_test == 1) & (y_pred == 1)).sum())
legitimas_reales = int((y_test == 0).sum())
legitimas_detectadas = int(((y_test == 0) & (y_pred == 0)).sum())

print(f"\n⚠️  Transacciones Fraudulentas:")
print(f"  - Total reales en el test: {fraudes_reales}")
print(f"  - Detectadas correctamente: {fraudes_detectados}")

print(f"\n💳 Transacciones Legítimas:")
print(f"  - Total reales en el test: {legitimas_reales}")
print(f"  - Detectadas correctamente: {legitimas_detectadas}")

# 8. Gráfico combinado
reales_legitimas = legitimas_reales
reales_fraude = fraudes_reales
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
plt.savefig(os.path.join(output_dir, "grafico_comparativo_transacciones.png"))

print("\n📊 Gráfico combinado guardado como 'grafico_comparativo_transacciones.png'")
