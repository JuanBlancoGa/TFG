import pandas as pd

# Cargar el archivo Excel y la hoja correcta
df = pd.read_excel("Hidrografia San Esteban.xlsx", sheet_name="Datos")

# Mostrar las columnas originales
print("Columnas originales detectadas:", list(df.columns))

# Asignar nombres correctos si hay 6 columnas
if len(df.columns) == 6:
    df.columns = [
        "Fecha",
        "Nivel_Embalse_msnm",
        "Caudal_Salida_m3s",
        "Precipitacion_mm",
        "Caudal_Entrada_m3s",
        "Comentario"  # Puedes cambiar este nombre según lo que contenga
    ]
else:
    raise ValueError("Número inesperado de columnas: se esperaban 6.")

# Convertir la columna de fecha
df["Fecha"] = pd.to_datetime(df["Fecha"])

# Detectar y corregir anomalías en el caudal de entrada (> 2000)
entrada = df["Caudal_Entrada_m3s"]
anomalias_idx = df[entrada > 2000].index
valid_idx = entrada[entrada <= 2000].index

# Interpolación
for idx in anomalias_idx:
    lower_idx = idx - 1
    while lower_idx not in valid_idx and lower_idx >= 0:
        lower_idx -= 1
    upper_idx = idx + 1
    while upper_idx not in valid_idx and upper_idx < len(entrada):
        upper_idx += 1
    if lower_idx in valid_idx and upper_idx in valid_idx:
        lower_val = entrada[lower_idx]
        upper_val = entrada[upper_idx]
        interpolated = lower_val + (upper_val - lower_val) * (
            (idx - lower_idx) / (upper_idx - lower_idx)
        )
        df.at[idx, "Caudal_Entrada_m3s"] = interpolated
    elif lower_idx in valid_idx:
        df.at[idx, "Caudal_Entrada_m3s"] = entrada[lower_idx]
    elif upper_idx in valid_idx:
        df.at[idx, "Caudal_Entrada_m3s"] = entrada[upper_idx]

# Guardar el archivo corregido
df.to_excel("Caudal_Corregido_Ospeares.xlsx", index=False)
