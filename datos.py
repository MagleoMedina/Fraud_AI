import numpy as np
import pandas as pd

def generar_dataset_csv(n_muestras=300, seed=42, ruta='dataset.csv'):
    np.random.seed(seed)

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

    # Guardar como CSV
    df.to_csv(ruta, index=False)
    print(f"✅ Dataset generado y guardado como '{ruta}' con {len(df)} registros (incluye sobremuestreo de fraudes).")

# Ejecutar directamente si se corre como script
if __name__ == "__main__":
    generar_dataset_csv()
