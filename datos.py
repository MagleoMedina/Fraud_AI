import numpy as np
import pandas as pd

def generar_dataset_csv(n_muestras=600, seed=42, ruta='dataset2.csv'):
    np.random.seed(seed)

    # Generar datos simulados
    monto = np.random.uniform(1, 1000, n_muestras)
    tipo_transaccion = np.random.choice([0, 1], n_muestras)  # 0: compra online, 1: transferencia
    hora = np.random.uniform(0, 24, n_muestras)
    ubicacion = np.random.uniform(0, 100, n_muestras)
    frecuencia = np.random.randint(1, 30, n_muestras)
    dispositivo_nuevo = np.random.choice([0, 1], n_muestras, p=[0.4, 0.6])  # ahora 60% chance dispositivo nuevo

    # Regla básica para clasificar fraude con mayor probabilidad
    is_fraud = ((monto > 500) & (dispositivo_nuevo == 1) & (frecuencia < 10)).astype(int)

    # Crear DataFrame
    df = pd.DataFrame({
        'monto': monto,
        'tipo_transaccion': tipo_transaccion,
        'hora': hora,
        'ubicacion': ubicacion,
        'frecuencia': frecuencia,
        'dispositivo_nuevo': dispositivo_nuevo,
        'isFraud': is_fraud
    })

    # Guardar como CSV
    df.to_csv(ruta, index=False)
    print(f"✅ Dataset generado y guardado como '{ruta}' con {n_muestras} registros.")

if __name__ == "__main__":
    generar_dataset_csv()
