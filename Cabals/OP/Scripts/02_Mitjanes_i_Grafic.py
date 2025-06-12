
import pandas as pd
import matplotlib.pyplot as plt

# Carregar dades corregides
df = pd.read_excel("Caudal_Corregido.xlsx")
df["Fecha"] = pd.to_datetime(df["Fecha"])

# Crear columna de mes i dia
df["Mes"] = df["Fecha"].dt.month
df["Día"] = df["Fecha"].dt.day

# Calcular mitjanes
media_diaria = df.groupby(["Mes", "Día"])[["Caudal_Entrada_m3s", "Caudal_Salida_m3s"]].mean().reset_index()
media_diaria["Fecha"] = pd.to_datetime({
    "year": 2000,
    "month": media_diaria["Mes"],
    "day": media_diaria["Día"]
})
media_diaria = media_diaria.sort_values("Fecha")

import matplotlib.dates as mdates

# Gràfic (formato del eje X solo con el mes)
plt.figure(figsize=(14, 6))
plt.plot(media_diaria["Fecha"], media_diaria["Caudal_Entrada_m3s"], label="Caudal d'entrada")
plt.plot(media_diaria["Fecha"], media_diaria["Caudal_Salida_m3s"], label="Caudal de sortida")
plt.title("Mitjana diària dels cabals")
plt.xlabel("Mes")
plt.ylabel("Cabal (m³/s)")
plt.legend()
plt.grid(True)

# Formatear eje X para que muestre solo el mes
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # '%b' = abreviatura del mes (Jan, Feb, ...)
# También puedes usar '%m' si prefieres solo el número del mes

plt.tight_layout()
plt.show()
media_diaria.to_excel("Media_Diaria_Caudales.xlsx", index=False)
