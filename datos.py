import os
import numpy as np
import pandas as pd

def generar_dataset_csv(n_muestras=300, seed=42, ruta_base='dataset/dataset'):
    np.random.seed(seed)

    # Crear carpeta 'dataset' si no existe
    dataset_dir = "dataset"
    os.makedirs(dataset_dir, exist_ok=True)

    # Generar datos simulados
    monto = np.random.uniform(1, 1000, n_muestras)
    tipo_transaccion = np.random.choice([0, 1], n_muestras)  # 0: compra online, 1: transferencia
    hora = np.random.uniform(0, 24, n_muestras)
    ubicacion = np.random.uniform(0, 100, n_muestras)
    frecuencia = np.random.randint(1, 30, n_muestras)
    dispositivo_nuevo = np.random.choice([0, 1], n_muestras)

    # Regla modificada para generar más fraudes
    is_fraud = ((monto > 400) & (dispositivo_nuevo == 1) & (frecuencia < 15)).astype(int)

    # Crear DataFrame original
    df = pd.DataFrame({
        'monto': monto,
        'tipo_transaccion': tipo_transaccion,
        'hora': hora,
        'ubicacion': ubicacion,
        'frecuencia': frecuencia,
        'dispositivo_nuevo': dispositivo_nuevo,
        'isFraud': is_fraud
    })

    # Sobremuestreo manual de fraudes para mejorar el balance
    df_fraude = df[df["isFraud"] == 1]
    if not df_fraude.empty:
        df = pd.concat([df, df_fraude.sample(frac=3, replace=True, random_state=seed)], ignore_index=True)

    # Dividir el dataset en 60% y 40%
    df_60 = df.sample(frac=0.6, random_state=seed)
    df_40 = df.drop(df_60.index)

    # Guardar como CSV
    df_60.to_csv(f"{ruta_base}60.csv", index=False)
    df_40.to_csv(f"{ruta_base}40.csv", index=False)
    print(f"✅ Dataset dividido y guardado como '{ruta_base}60.csv' ({len(df_60)} registros) y '{ruta_base}40.csv' ({len(df_40)} registros).")

# Ejecutar directamente si se corre como script
if __name__ == "__main__":
    generar_dataset_csv()
