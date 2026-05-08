
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')
 
 print("="*70)
 print("ANÁLISIS DE SPOTIFY - REGRESIÓN LINEAL MÚLTIPLE")
 print("="*70)
 
 # 1. Cargar el dataset con encoding latin1
 print("\n1. CARGANDO EL DATASET...")
 try:
     df = pd.read_csv(r'C:\Users\Emilio\Desktop\Exceles\Spotify.csv', encoding='latin1')
     print("✓ Dataset cargado exitosamente")
     print(f"  Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
 except FileNotFoundError:
     print("✗ Error: No se encontró el archivo 'Spotify.csv'")
     exit()
 
 print("\nPrimeras filas del dataset:")
 print(df.head())
 print("\nColumnas disponibles:")
 print(df.columns.tolist())
 
 # 2. Convertir columnas booleanas de True/False a 0/1
 print("\n2. CONVIRTIENDO COLUMNAS BOOLEANAS...")
 df['shuffle'] = df['shuffle'].astype(int)
 df['skipped'] = df['skipped'].astype(int)
 print("✓ Columnas convertidas exitosamente")
 print("\nMuestra de datos después de la conversión:")
 print(df[['shuffle', 'skipped', 'ms_played']].head(10))
 
 # 3. Ajustar el modelo de regresión lineal múltiple
 print("\n3. AJUSTANDO EL MODELO DE REGRESIÓN LINEAL MÚLTIPLE...")
 modelo = smf.ols('ms_played ~ shuffle + skipped', data=df).fit()
 print("✓ Modelo ajustado exitosamente")
 
 print("\nRESUMEN DEL MODELO:")
 print(modelo.summary())
 
 # 4. Mostrar la ecuación del modelo
 print("\n" + "="*70)
 print("ECUACIÓN DEL MODELO DE REGRESIÓN LINEAL MÚLTIPLE")
 print("="*70)
 
 intercept = modelo.params['Intercept']
 coef_shuffle = modelo.params['shuffle']
 coef_skipped = modelo.params['skipped']
 
 print(f"\nms_played = {intercept:.6f} + {coef_shuffle:.6f}*shuffle + {coef_skipped:.6f}*skipped")
 print(f"\nms_played = {intercept:.2f} + {coef_shuffle:.2f}*shuffle + {coef_skipped:.2f}*skipped")
 
 print("\nInterpretación de los coeficientes:")
 print(f"  • Intercept (β0): {intercept:.2f} ms (ms_played predicho cuando shuffle=0 y skipped=0)")
 print(f"  • Coeficiente de shuffle (β1): {coef_shuffle:.2f} ms")
 print(f"  • Coeficiente de skipped (β2): {coef_skipped:.2f} ms")
 
 print(f"\nMétricas de ajuste:")
 print(f"  • R-squared: {modelo.rsquared:.6f}")
 print(f"  • R-squared ajustado: {modelo.rsquared_adj:.6f}")
 
 # 5. Hacer predicciones para casos específicos
 print("\n" + "="*70)
 print("PREDICCIONES PARA CASOS ESPECÍFICOS")
 print("="*70)
 
 casos = pd.DataFrame({
     'shuffle': [0, 1, 0],
     'skipped': [0, 0, 1]
 })
 
 predicciones = modelo.predict(casos)
 
 print("\nCasos analizados:")
 for i, (idx, row) in enumerate(casos.iterrows(), 1):
     pred = predicciones.iloc[i-1]
     print(f"\nCaso {i}: shuffle = {int(row['shuffle'])}, skipped = {int(row['skipped'])}")
     print(f"  Predicción de ms_played = {pred:.2f} ms")
 
 # Tabla resumen
 resultado = casos.copy()
 resultado['ms_played_predicho'] = predicciones.values
 print("\n" + "="*70)
 print("TABLA RESUMEN DE PREDICCIONES")
 print("="*70)
 print(resultado.to_string(index=False))

# 6. Calcular residuales
print("\n" + "="*70)
print("ANÁLISIS DE RESIDUALES")
print("="*70)

# Calcular predicciones para todos los datos
predicciones_completas = modelo.predict(df[['shuffle', 'skipped']])

# Calcular residuales: residual = Y_real - Y_predicho
residuales = df['ms_played'] - predicciones_completas

print(f"\nTotal de registros analizados: {len(residuales)}")

# Residual más pequeño
residual_min = residuales.min()
print(f"Residual más pequeño: {residual_min:.2f} ms")

# Residual más grande
residual_max = residuales.max()
print(f"Residual más grande: {residual_max:.2f} ms")

# Contar residuales negativos
residuales_negativos = (residuales < 0).sum()
print(f"Número de residuales negativos: {residuales_negativos}")

# Desviación estándar de los residuales
residual_std = residuales.std()
print(f"Desviación estándar de los residuales: {residual_std:.2f} ms")

# Estadísticas adicionales de los residuales
print("Estadísticas descriptivas de los residuales:")
print(residuales.describe())
 
 print("\n" + "="*70)
 print("ANÁLISIS COMPLETADO")
 print("="*70)