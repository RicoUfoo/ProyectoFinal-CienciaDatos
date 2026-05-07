import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
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

# Verificar niveles de significancia
p_shuffle = modelo.pvalues['shuffle']
p_skipped = modelo.pvalues['skipped']
F_stat = modelo.fvalue
F_pvalue = modelo.f_pvalue
adj_r2 = modelo.rsquared_adj

print("\nVERIFICACIÓN DE SIGNIFICANCIA")
print(f"  • p-value de shuffle: {p_shuffle:.6g} -> {'significativo' if p_shuffle < 0.05 else 'no significativo'}")
print(f"  • p-value de skipped: {p_skipped:.6g} -> {'significativo' if p_skipped < 0.05 else 'no significativo'}")
print(f"  • F-statistic: {F_stat:.4f}")
print(f"  • p-value del F-statistic: {F_pvalue:.6g} -> {'significativo' if F_pvalue < 0.05 else 'no significativo'}")
print(f"  • R-squared ajustado: {adj_r2:.6f}")

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

r2_percent = modelo.rsquared * 100
print(f"\nMétricas de ajuste:")
print(f"  • R-squared: {modelo.rsquared:.6f}")
print(f"  • R-squared ajustado: {modelo.rsquared_adj:.6f}")
print(f"  • R-squared en porcentaje: {r2_percent:.4f}%")

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

# 6. Calcular la matriz de correlación de Pearson
print("\n" + "="*70)
print("MATRIZ DE CORRELACIÓN DE PEARSON")
print("="*70)

correlacion = df[['shuffle', 'skipped', 'ms_played']].corr(method='pearson')
print(correlacion)

corr_shuffle = correlacion.loc['shuffle', 'ms_played']
corr_skipped = correlacion.loc['skipped', 'ms_played']
corr_x1_x2 = correlacion.loc['shuffle', 'skipped']

print("\nCorrelación con ms_played:")
print(f"  • shuffle vs ms_played: {corr_shuffle:.4f}")
print(f"  • skipped vs ms_played: {corr_skipped:.4f}")

if abs(corr_shuffle) > abs(corr_skipped):
    print("  -> shuffle tiene mayor correlación con ms_played.")
elif abs(corr_skipped) > abs(corr_shuffle):
    print("  -> skipped tiene mayor correlación con ms_played.")
else:
    print("  -> Ambas variables tienen correlaciones de igual magnitud con ms_played.")

print(f"\nCorrelación entre shuffle y skipped: {corr_x1_x2:.4f}")
if abs(corr_x1_x2) < 0.1:
    print("  -> Prácticamente no hay correlación lineal entre shuffle y skipped.")
elif abs(corr_x1_x2) < 0.5:
    print("  -> Existe una correlación lineal moderada entre shuffle y skipped.")
else:
    print("  -> Hay una correlación lineal fuerte entre shuffle y skipped.")

# 7. Calcular residuales
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

# Calcular Durbin-Watson
dw_stat = durbin_watson(residuales)
dw_ok = 1.5 <= dw_stat <= 2.5
print(f"Durbin-Watson: {dw_stat:.4f}")
print(f"  -> Valor entre 1.5 y 2.5: {'sí' if dw_ok else 'no'}")
print(f"  -> Independencia de residuales: {'se cumple' if dw_ok else 'no se cumple'}")

# Prueba de Breusch-Pagan para homocedasticidad
bp_test = het_breuschpagan(residuales, modelo.model.exog)
bp_pvalue = bp_test[1]
print(f"Breusch-Pagan p-value: {bp_pvalue:.6g}")
print(f"  -> {'homocedasticidad' if bp_pvalue > 0.05 else 'heterocedasticidad'}")

# Calcular RMSE
rmse_ms = np.sqrt(np.mean(residuales**2))
rmse_s = rmse_ms / 1000.0
mean_ms_played = df['ms_played'].mean()
rmse_pct_mean = (rmse_ms / mean_ms_played) * 100

print(f"RMSE: {rmse_ms:.2f} ms")
print(f"RMSE: {rmse_s:.2f} s")
print(f"Media real de ms_played: {mean_ms_played:.2f} ms")
print(f"RMSE como porcentaje de la media: {rmse_pct_mean:.2f}%")

# Estadísticas adicionales de los residuales
print("\nEstadísticas descriptivas de los residuales:")
print(residuales.describe())

# 8. Generar dashboard de gráficos
print("\n" + "="*70)
print("DASHBOARD DE GRÁFICOS")
print("="*70)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

axes[0, 0].scatter(df['shuffle'], df['ms_played'], alpha=0.3, s=10)
axes[0, 0].set_xlabel('shuffle')
axes[0, 0].set_ylabel('ms_played')
axes[0, 0].set_title('Dispersion shuffle vs ms_played')
axes[0, 0].grid(True)

axes[0, 1].scatter(df['skipped'], df['ms_played'], alpha=0.3, s=10)
axes[0, 1].set_xlabel('skipped')
axes[0, 1].set_ylabel('ms_played')
axes[0, 1].set_title('Dispersion skipped vs ms_played')
axes[0, 1].grid(True)

axes[1, 0].scatter(predicciones_completas, residuales, alpha=0.3, s=10)
axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=1)
axes[1, 0].set_xlabel('ms_played predicho')
axes[1, 0].set_ylabel('Residuales')
axes[1, 0].set_title('Residuales vs Predichos')
axes[1, 0].grid(True)

sm.qqplot(residuales, line='45', ax=axes[1, 1])
axes[1, 1].set_title('QQ-plot de los residuales')

plt.tight_layout()
plt.savefig('dashboard_TUMATRICULA.png', dpi=150, bbox_inches='tight')
print("Dashboard guardado en dashboard_TUMATRICULA.png")

print("\n" + "="*70)
print("ANÁLISIS COMPLETADO")
print("="*70)