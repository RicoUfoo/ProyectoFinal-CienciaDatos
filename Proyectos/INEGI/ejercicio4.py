import pandas as pd

# Lee solo la hoja correcta
df = pd.read_excel("Ejercicio2_inegi.xlsx", sheet_name="denue_inegi_61_")

# Tabla de conversion de per_ocu a numerico
conversion = {
    "0 a 5 personas": 3,
    "6 a 10 personas": 8,
    "11 a 30 personas": 20,
    "31 a 50 personas": 40,
    "51 a 100 personas": 75,
    "101 a 250 personas": 175,
    "251 y más personas": 251
}

# Convertir per_ocu a numero
df["per_ocu_numeric"] = df["per_ocu"].map(conversion)

# Eliminar filas con valores vacios en latitud, longitud o per_ocu
df = df.dropna(subset=["latitud", "longitud", "per_ocu_numeric"])

# Guardar archivo limpio
df.to_excel("Ejercicio4_EmilioZuñiga_AL07182673.xlsx", index=False)

print("=" * 40)
print("Limpieza completada")
print(f"Filas originales: 140,666")
print(f"Filas limpias: {len(df)}")
print(f"Columnas: {df.shape[1]}")
print()
print("Muestra de per_ocu_numeric:")
print(df["per_ocu_numeric"].value_counts())