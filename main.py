import numpy as np
import os
import matplotlib.pyplot as plt
import sympy as sp
import pandas as pd
from scipy.optimize import fsolve
from sympy import *
from sympy.plotting import plot
from sympy.abc import x

from funciones import newton_raphson, rp, areaseg, area1, area2, ra, phiuno, phidos

def comparar_curvas(i, q, r1, r2, L, P, nombre_archivo): # Función para analisar las curvas
    R = 1
    l1 = 1
    l2 = L
    fase = np.linspace(0, 1, 600) # 600 puntos
    theta = 2 * np.pi * fase
    y_values = []

    for th in theta:
        x = R * np.sin(th)
        y = R * np.cos(i) * np.cos(th)
        z = R * np.sin(i) * np.cos(th)

        x1 = -x / (1 + (1 / q))
        y1 = -y / (1 + (1 / q))
        z1 = -z / (1 + (1 / q))

        x2 = x / (1 + q)
        y2 = y / (1 + q)
        z2 = z / (1 + q)

        ro = ra(x1, x2, y1, y2)
        phi1 = phiuno(r1, r2, ro)
        phi2 = phidos(r1, r2, ro)
        a1 = area1(ro, r1, r2, z1, z2, phi1, phi2)
        a2 = area2(ro, r1, r2, z1, z2, phi1, phi2)

        y_val = (l1 * a1 / r1**2 + l2 * a2 / r2**2)
        y_values.append(y_val)

    y_values = np.array(y_values)  # Valores de la curva sintética
    y_values_guardado = np.array(y_values)

    index_025 = np.argmin(np.abs(fase - 0.25))
    index_075 = np.argmin(np.abs(fase - 0.75))
    Lmax = (y_values_guardado[index_025] + y_values_guardado[index_075]) / 2

    if np.isnan(Lmax) or Lmax == 0:
        print("Error: Lmax no es válido.")
        return

    ruta_base = os.path.dirname(__file__)  # Carpeta donde está la data
    archivo_txt = os.path.join(ruta_base, 'data', f'{nombre_archivo}.txt')

    if not os.path.exists(archivo_txt):   # Verificar si el archivo existe
        print(f"El archivo {archivo_txt} no existe.")
        return

    datos = pd.read_csv(archivo_txt, delimiter='\t', header=0) # Leer los datos desde el archivo de texto
    magnitud = datos['MAG']
    tiempos = datos['HJD']

    # Filtrar datos no válidos
    datos_validos = datos.dropna(subset=['MAG', 'HJD'])  # Eliminar filas con valores NaN (evitar errores de lectura)
    datos_validos = datos_validos[
        ~np.isinf(datos_validos['MAG']) & ~np.isinf(datos_validos['HJD'])
    ]  # Eliminar infinitos
    datos_validos = datos_validos[
        datos_validos['MAG'].apply(np.isreal) & datos_validos['HJD'].apply(np.isreal)
    ]  # Verificar que sean números reales

    if datos_validos.empty:
        print("Error: No quedan datos válidos después de la depuración.")
        return

    magnitud = datos_validos['MAG']
    tiempos = datos_validos['HJD']

    top_10_indices = magnitud.nlargest(10).index
    top_10_tiempos = tiempos.loc[top_10_indices]

    errores_cuadraticos_totales = []

    for t0 in top_10_tiempos: #Para los 10 flujos más grandes
        nuevos_tiempos = (tiempos - t0) / P - np.floor((tiempos - t0) / P)
        flujo = 10 ** (-magnitud / 2.5) * 10 ** 4
        datos_fase = pd.DataFrame({
            'Tiempo en fase': nuevos_tiempos,
            'Magnitud': flujo
        })
        datos_fase_ordenados = datos_fase.sort_values(by='Tiempo en fase')

        magnitudes_fase_025 = datos_fase_ordenados[(datos_fase_ordenados['Tiempo en fase'] >= 0.24) &
                                                   (datos_fase_ordenados['Tiempo en fase'] <= 0.26)]
        magnitudes_fase_075 = datos_fase_ordenados[(datos_fase_ordenados['Tiempo en fase'] >= 0.74) &
                                                   (datos_fase_ordenados['Tiempo en fase'] <= 0.76)]
        promedio_025 = magnitudes_fase_025['Magnitud'].mean()
        promedio_075 = magnitudes_fase_075['Magnitud'].mean()
        factor_escalar = (promedio_025 + promedio_075) / 2

        if np.isnan(factor_escalar) or factor_escalar == 0:
            continue

        y_values_normalizados = (y_values / Lmax) * factor_escalar

        error_cuadratico_total = 0
        errores_individuales = []
        errores_fase = []

        for idx, row in datos_fase_ordenados.iterrows(): #Se busca el caso que mejor ajuste
            fase_cercano = np.argmin(np.abs(fase - row['Tiempo en fase']))
            y_teorico = y_values_normalizados[fase_cercano]
            y_experimental = row['Magnitud']
            error = y_experimental - y_teorico
            error_cuadratico_total += error ** 2

            errores_individuales.append(error)
            errores_fase.append(fase[fase_cercano])

        if np.isnan(error_cuadratico_total) or np.isinf(error_cuadratico_total):
            continue

        errores_cuadraticos_totales.append((error_cuadratico_total, t0, datos_fase_ordenados, factor_escalar, errores_individuales, errores_fase))

    if len(errores_cuadraticos_totales) == 0:
        print("Error: No se encontraron casos válidos después del análisis.")
        return

    # Seleccionar el mejor caso
    mejor_error, mejor_t0, mejor_datos_fase_ordenados, mejor_factor_escalar, mejor_errores_individuales, mejor_errores_fase = min(errores_cuadraticos_totales, key=lambda x: x[0])
    mejor_error = mejor_error / len(mejor_datos_fase_ordenados['Magnitud'])

    # Información de las curvas
    print(f"Mejor t0 seleccionado: {mejor_t0}")
    print(f"Numero de puntos experimentales= {len(mejor_datos_fase_ordenados['Magnitud'])}")
    print(f"Promedio de magnitudes en fase 0.25: {promedio_025}")
    print(f"Promedio de magnitudes en fase 0.75: {promedio_075}")
    print(f"Factor escalar = {mejor_factor_escalar}")
    print(f"Suma de errores cuadráticos normalizada: {mejor_error}")

    y_values_normalizados = (y_values_guardado / Lmax) * mejor_factor_escalar  # Curva sintética escalada

    plt.figure() # Graficar
    plt.plot(fase, y_values_normalizados, 'o', markersize=1.5, color='black', label='Valores teóricos')
    plt.plot(mejor_datos_fase_ordenados['Tiempo en fase'], mejor_datos_fase_ordenados['Magnitud'],
             'x', markersize=5, markeredgewidth=1, label='Valores observados')
    plt.plot(mejor_errores_fase, mejor_errores_individuales, 'rx', markersize=3, label='Errores')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    plt.xlim(0, 1)
    plt.xlabel('Fase')
    plt.ylabel('Flujo Relativo (x10^-4)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3)
    plt.tight_layout()
    plt.grid(True)
    plt.show()


def comparar_curvas_pot(i, q, p1, p2, L, P, nombre_archivo): # Función para analisar las curvas con parametro de potencial
    R = 1
    l1 = 1
    l2 = L
    r1= rp(q,p1)
    r2= rp(q,p2)
    fase = np.linspace(0, 1, 600) # 600 puntos
    theta = 2 * np.pi * fase
    y_values = []

    for th in theta:
        x = R * np.sin(th)
        y = R * np.cos(i) * np.cos(th)
        z = R * np.sin(i) * np.cos(th)

        x1 = -x / (1 + (1 / q))
        y1 = -y / (1 + (1 / q))
        z1 = -z / (1 + (1 / q))

        x2 = x / (1 + q)
        y2 = y / (1 + q)
        z2 = z / (1 + q)

        ro = ra(x1, x2, y1, y2)
        phi1 = phiuno(r1, r2, ro)
        phi2 = phidos(r1, r2, ro)
        a1 = area1(ro, r1, r2, z1, z2, phi1, phi2)
        a2 = area2(ro, r1, r2, z1, z2, phi1, phi2)

        y_val = (l1 * a1 / r1**2 + l2 * a2 / r2**2)
        y_values.append(y_val)

    y_values = np.array(y_values)  # Valores de la curva sintética
    y_values_guardado = np.array(y_values)

    index_025 = np.argmin(np.abs(fase - 0.25))
    index_075 = np.argmin(np.abs(fase - 0.75))
    Lmax = (y_values_guardado[index_025] + y_values_guardado[index_075]) / 2

    if np.isnan(Lmax) or Lmax == 0:
        print("Error: Lmax no es válido.")
        return

    ruta_base = os.path.dirname(__file__)  # Carpeta donde está la data
    archivo_txt = os.path.join(ruta_base, 'data', f'{nombre_archivo}.txt')

    if not os.path.exists(archivo_txt):   # Verificar si el archivo existe
        print(f"El archivo {archivo_txt} no existe.")
        return

    datos = pd.read_csv(archivo_txt, delimiter='\t', header=0) # Leer los datos desde el archivo de texto
    magnitud = datos['MAG']
    tiempos = datos['HJD']

    # Filtrar datos no válidos
    datos_validos = datos.dropna(subset=['MAG', 'HJD'])  # Eliminar filas con valores NaN (evitar errores de lectura)
    datos_validos = datos_validos[
        ~np.isinf(datos_validos['MAG']) & ~np.isinf(datos_validos['HJD'])
    ]  # Eliminar infinitos
    datos_validos = datos_validos[
        datos_validos['MAG'].apply(np.isreal) & datos_validos['HJD'].apply(np.isreal)
    ]  # Verificar que sean números reales

    if datos_validos.empty:
        print("Error: No quedan datos válidos después de la depuración.")
        return

    magnitud = datos_validos['MAG']
    tiempos = datos_validos['HJD']

    top_10_indices = magnitud.nlargest(10).index
    top_10_tiempos = tiempos.loc[top_10_indices]

    errores_cuadraticos_totales = []

    for t0 in top_10_tiempos: #Para los 10 flujos más grandes
        nuevos_tiempos = (tiempos - t0) / P - np.floor((tiempos - t0) / P)
        flujo = 10 ** (-magnitud / 2.5) * 10 ** 4
        datos_fase = pd.DataFrame({
            'Tiempo en fase': nuevos_tiempos,
            'Magnitud': flujo
        })
        datos_fase_ordenados = datos_fase.sort_values(by='Tiempo en fase')

        magnitudes_fase_025 = datos_fase_ordenados[(datos_fase_ordenados['Tiempo en fase'] >= 0.24) &
                                                   (datos_fase_ordenados['Tiempo en fase'] <= 0.26)]
        magnitudes_fase_075 = datos_fase_ordenados[(datos_fase_ordenados['Tiempo en fase'] >= 0.74) &
                                                   (datos_fase_ordenados['Tiempo en fase'] <= 0.76)]
        promedio_025 = magnitudes_fase_025['Magnitud'].mean()
        promedio_075 = magnitudes_fase_075['Magnitud'].mean()
        factor_escalar = (promedio_025 + promedio_075) / 2

        if np.isnan(factor_escalar) or factor_escalar == 0:
            continue

        y_values_normalizados = (y_values / Lmax) * factor_escalar

        error_cuadratico_total = 0
        errores_individuales = []
        errores_fase = []

        for idx, row in datos_fase_ordenados.iterrows(): #Se busca el caso que mejor ajuste
            fase_cercano = np.argmin(np.abs(fase - row['Tiempo en fase']))
            y_teorico = y_values_normalizados[fase_cercano]
            y_experimental = row['Magnitud']
            error = y_experimental - y_teorico
            error_cuadratico_total += error ** 2

            errores_individuales.append(error)
            errores_fase.append(fase[fase_cercano])

        if np.isnan(error_cuadratico_total) or np.isinf(error_cuadratico_total):
            continue

        errores_cuadraticos_totales.append((error_cuadratico_total, t0, datos_fase_ordenados, factor_escalar, errores_individuales, errores_fase))

    if len(errores_cuadraticos_totales) == 0:
        print("Error: No se encontraron casos válidos después del análisis.")
        return

    # Seleccionar el mejor caso
    mejor_error, mejor_t0, mejor_datos_fase_ordenados, mejor_factor_escalar, mejor_errores_individuales, mejor_errores_fase = min(errores_cuadraticos_totales, key=lambda x: x[0])
    mejor_error = mejor_error / len(mejor_datos_fase_ordenados['Magnitud'])

    # Información de las curvas
    print(f"Mejor t0 seleccionado: {mejor_t0}")
    print(f"Numero de puntos experimentales= {len(mejor_datos_fase_ordenados['Magnitud'])}")
    print(f"Promedio de magnitudes en fase 0.25: {promedio_025}")
    print(f"Promedio de magnitudes en fase 0.75: {promedio_075}")
    print(f"Factor escalar = {mejor_factor_escalar}")
    print(f"Suma de errores cuadráticos normalizada: {mejor_error}")

    y_values_normalizados = (y_values_guardado / Lmax) * mejor_factor_escalar  # Curva sintética escalada

    plt.figure() # Graficar
    plt.plot(fase, y_values_normalizados, 'o', markersize=1.5, color='black', label='Valores teóricos')
    plt.plot(mejor_datos_fase_ordenados['Tiempo en fase'], mejor_datos_fase_ordenados['Magnitud'],
             'x', markersize=5, markeredgewidth=1, label='Valores observados')
    plt.plot(mejor_errores_fase, mejor_errores_individuales, 'rx', markersize=3, label='Errores')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    plt.xlim(0, 1)
    plt.xlabel('Fase')
    plt.ylabel('Flujo Relativo (x10^-4)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3)
    plt.tight_layout()
    plt.grid(True)
    plt.show()