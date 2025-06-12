
import pandas as pd

# Carregar el fitxer original
df = pd.read_excel("Hidrografia San Esteban.xlsx", sheet_name="Datos")

# Renombrar columnes
df.columns = [
    "Fecha",
    "Nivel_Embalse_msnm",
    "Caudal_Salida_m3s",
    "Precipitacion_mm",
    "Caudal_Entrada_m3s"
]

# Convertir la data
df["Fecha"] = pd.to_datetime(df["Fecha"])

# Trobar anomalies (>2000)
entrada = df["Caudal_Entrada_m3s"]
anomalias_idx = df[df["Caudal_Entrada_m3s"] > 2000].index
valid_idx = entrada[entrada <= 2000].index

# Interpolació
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

# Guardar dades corregides
df.to_excel("Caudal_Corregido.xlsx", index=False)
