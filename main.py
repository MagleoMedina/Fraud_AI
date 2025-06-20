import pandas as pd
from mlp import MLP
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import joblib
import numpy as np

# Crear la carpeta 'graphics' si no existe
output_dir = "graphics"
os.makedirs(output_dir, exist_ok=True)

# 1. Cargar dataset para prueba (40%) y omitir la columna isFraud
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
y_pred = mlp.predict(X_test)

#lo use para testear sensibilidad
#y_prob = mlp.predict_proba(X_test)
#y_pred = (y_prob >= 0.65).astype(int)  

# 6. Evaluar resultados
report_dict = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)
print(f"Exactitud del modelo sobre datos no vistos: {accuracy_score(y_test, y_pred) * 100:.2f}%")

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
print("")

# Crear gráfica de barras con los datos de fraudes y legítimas
fig, ax = plt.subplots(figsize=(8, 5))
categorias = ['Fraudes Reales', 'Fraudes Detectados', 'Legítimas Reales', 'Legítimas Detectadas']
valores = [fraudes_reales, fraudes_detectados, legitimas_reales, legitimas_detectadas]
colores = ['red', 'green', 'blue', 'purple']

ax.bar(categorias, valores, color=colores)
ax.set_title('Comparación de Transacciones')
ax.set_ylabel('Cantidad')
ax.set_xlabel('Categorías')
plt.xticks(rotation=45)

# Guardar gráfica en la carpeta 'graphics'
output_path = os.path.join(output_dir, "grafico_transacciones_barras.png")
plt.tight_layout()
plt.savefig(output_path)
plt.close()

print(f"📊 Gráfico de barras guardado como '{output_path}'")

# === Gráfica del reporte de clasificación ===

# Extraer métricas para clases 0 y 1
labels = ['Legítima (0)', 'Fraude (1)']
precision = [report_dict['0']['precision'], report_dict['1']['precision']]
recall = [report_dict['0']['recall'], report_dict['1']['recall']]
f1 = [report_dict['0']['f1-score'], report_dict['1']['f1-score']]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width, precision, width, label='Precisión')
rects2 = ax.bar(x, recall, width, label='Recall')
rects3 = ax.bar(x + width, f1, width, label='F1-score')

ax.set_ylabel('Valor')
ax.set_title('Reporte de Clasificación por Clase')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.2)
ax.legend()

for rects in [rects1, rects2, rects3]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "grafico_reporte_clasificacion.png"))
plt.close()

print(f"📊 Gráfico del reporte de clasificación guardado como '{os.path.join(output_dir, 'grafico_reporte_clasificacion.png')}'")

# === Gráfica de la matriz de confusión ===
fig, ax = plt.subplots(figsize=(5, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legítima", "Fraude"])
disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format='d')
ax.set_title("Matriz de Confusión")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "grafico_matriz_confusion.png"))
plt.close()

print(f"📊 Gráfico de la matriz de confusión guardado como '{os.path.join(output_dir, 'grafico_matriz_confusion.png')}'")

# Cargar métricas de entrenamiento
metricas_path = os.path.join("metrics", "metricas_entrenamiento.npz")
if os.path.exists(metricas_path):
    datos_metricas = np.load(metricas_path, allow_pickle=True)
    loss_history = datos_metricas['loss_history']
    metrics_history = datos_metricas['metrics_history']
else:
    raise FileNotFoundError(f"No se encontró el archivo de métricas '{metricas_path}'. Entrena primero el modelo.")

# Graficar curvas de aprendizaje (Loss y Accuracy)
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Loss
ax[0].plot(loss_history, label="Loss", color="blue")
ax[0].set_title("Curva de Pérdida")
ax[0].set_xlabel("Época")
ax[0].set_ylabel("Pérdida")
ax[0].legend()

# Accuracy
accuracy_history = [m['accuracy'] for m in metrics_history]
ax[1].plot(accuracy_history, label="Exactitud", color="green")
ax[1].set_title("Curva de Exactitud")
ax[1].set_xlabel("Época")
ax[1].set_ylabel("Exactitud")
ax[1].legend()

# Guardar gráfica en la carpeta 'graphics'
output_path = os.path.join(output_dir, "curvas_aprendizaje.png")
plt.tight_layout()
plt.savefig(output_path)
plt.close()

print(f"📊 Gráfico de curvas de aprendizaje guardado como '{output_path}'")
