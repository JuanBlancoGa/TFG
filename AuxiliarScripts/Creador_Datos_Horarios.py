import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Leer los datos desde el archivo Excel
# Asegúrate de que 'datos.xlsx' esté en el mismo directorio que este script
# Usa header=None si el Excel no tiene encabezados
original_df = pd.read_excel('Datos.xlsx', header=None)

# Confirmamos dimensiones
if original_df.shape != (7, 8):
    raise ValueError("El archivo debe tener exactamente 7 filas y 8 columnas.")

# Número de puntos a interpolar entre cada fila
steps_per_day = 24
total_steps = (original_df.shape[0] - 1) * steps_per_day  # 6 * 24 = 144

# Índices originales (0 a 6)
x_original = np.arange(original_df.shape[0])

# Nuevos índices interpolados: 0, 1/24, ..., 5 + 23/24
x_interp = np.linspace(0, original_df.shape[0] - 1, total_steps)

# Interpolación por columna
interpolated_data = {}
for i, col in enumerate(original_df.columns):
    f = interp1d(x_original, original_df[col], kind='linear')  # O 'cubic' para más suavidad
    interpolated_data[f'Col{i+1}'] = f(x_interp)

# Crear DataFrame interpolado de 168 filas (7*24)
interpolated_df = pd.DataFrame(interpolated_data)

# Guardar a nuevo Excel si quieres
interpolated_df.to_excel('datos_interpolados.xlsx', index=False)

print("Interpolación completada y guardada como 'datos_interpolados.xlsx'")
