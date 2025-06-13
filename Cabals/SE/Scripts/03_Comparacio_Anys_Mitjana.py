
import pandas as pd
import matplotlib.pyplot as plt

# Carregar dades
df = pd.read_excel("Caudal_Corregido.xlsx")
media_diaria = pd.read_excel("Media_Diaria_Caudales.xlsx")

df["Fecha"] = pd.to_datetime(df["Fecha"])
media_diaria["Fecha"] = pd.to_datetime(media_diaria["Fecha"])

df["Año"] = df["Fecha"].dt.year
df["Dia_Año"] = df["Fecha"].dt.dayofyear
media_diaria["Dia_Año"] = media_diaria["Fecha"].dt.dayofyear

anys = [2015, 2017, 2019, 2022, 2024]

for any in anys:
    df_any = df[df["Año"] == any]
    serie = df_any.groupby("Dia_Año")["Caudal_Entrada_m3s"].mean()

    plt.figure(figsize=(14, 5))
    plt.plot(media_diaria["Dia_Año"], media_diaria["Caudal_Entrada_m3s"], label="Mitjana (2015-2024)", linestyle="--")
    plt.plot(serie.index, serie.values, label=f"Any {any}")
    plt.title(f"Comparació del cabal d'entrada - Any {any}")
    plt.xlabel("Dia de l'any")
    plt.ylabel("Cabal d'entrada (m³/s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# Gràfica final amb 2015, 2019 i la mitjana, i guardar com PNG
anys_final = [2015, 2019]
plt.figure(figsize=(14, 5))

# Dibuixar la mitjana
plt.plot(media_diaria["Dia_Año"], media_diaria["Caudal_Entrada_m3s"], label="Mitjana (2015–2024)", linestyle="--")

# Dibuixar cada any sol·licitat
for any in anys_final:
    df_any = df[df["Año"] == any]
    serie = df_any.groupby("Dia_Año")["Caudal_Entrada_m3s"].mean()
    plt.plot(serie.index, serie.values, label=f"Any {any}")

plt.title("Comparació del cabal d'entrada - 2015 vs 2019 vs Mitjana")
plt.xlabel("Dia de l'any")
plt.ylabel("Cabal d'entrada (m³/s)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Guardar la figura com PNG
plt.savefig("Comparacio_Cabal_Entrada_2015_2019_Mitjana.png", dpi=300)

plt.show()
