"""ZONA FIJA: carga el modelo, calcula la probabilidad, aplica la regla de decisión y explica cada predicción.
No modifiquen este archivo: check.py verifica que siga igual."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_MODELO = RAIZ / "model"


def cargar_config():
    """Lee los datos de la app que el equipo completó en mi_app.py."""
    import importlib
    import sys
    sys.path.insert(0, str(RAIZ))
    import mi_app
    importlib.reload(mi_app)
    return {"titulo": mi_app.TITULO, "cliente": mi_app.CLIENTE, "descripcion": mi_app.DESCRIPCION,
            "unidad": mi_app.UNIDAD, "pie": mi_app.PIE, "logo": mi_app.LOGO, "interfaz": mi_app.INTERFAZ}


def hay_modelo():
    """True si ya exportaron el modelo a la carpeta model/."""
    return (CARPETA_MODELO / "pipeline.joblib").exists() and (CARPETA_MODELO / "metadata.json").exists()


def cargar_metadata():
    """Lee model/metadata.json, el archivo que escribe exportar.py."""
    with open(CARPETA_MODELO / "metadata.json", encoding="utf-8") as f:
        return json.load(f)


def cargar_modelo():
    """Carga model/pipeline.joblib, el modelo que entrenaron en Colab."""
    return joblib.load(CARPETA_MODELO / "pipeline.joblib")


def cargar_resultados():
    """Lee model/resultados.csv si existe (la lista, el radar o el mapa precalculado)."""
    ruta = CARPETA_MODELO / "resultados.csv"
    if not ruta.exists():
        return None
    return pd.read_csv(ruta)


def columnas_de_entrada(meta):
    return [c["nombre"] for c in meta["columnas"]]


def etiqueta(meta, columna):
    """Nombre de negocio de una columna, si el equipo lo definió al exportar."""
    return meta.get("etiquetas", {}).get(columna, columna)


def predecir_probabilidad(modelo, meta, datos):
    """Probabilidad del evento (por ejemplo, abandono o atraso) para cada fila de datos."""
    X = datos[columnas_de_entrada(meta)]
    clases = list(modelo.classes_)
    indice = clases.index(_convertir_clase(meta["clase_positiva"], clases))
    return modelo.predict_proba(X)[:, indice]


def _convertir_clase(valor, clases):
    for c in clases:
        if str(c) == str(valor):
            return c
        try:
            if float(c) == float(valor):
                return c
        except (TypeError, ValueError):
            pass
    raise ValueError(f"La clase positiva {valor!r} no está entre las clases del modelo: {clases}")


def decidir(probabilidades, meta):
    """Convierte cada probabilidad en una decisión con los umbrales del equipo.
    Con umbrales [u1, u2] y decisiones [d0, d1, d2]: p < u1 → d0; u1 ≤ p < u2 → d1; p ≥ u2 → d2."""
    umbrales = np.array(meta["umbrales"], dtype=float)
    decisiones = meta["decisiones"]
    posiciones = np.searchsorted(umbrales, np.atleast_1d(probabilidades), side="right")
    return [decisiones[i] for i in posiciones]


PALABRAS_VACIAS = {"de","la","que","el","en","y","a","los","se","del","las","un","por","con","no","una","su","para","es","al","lo","como","más","mas","pero","sus","le","ya","o","me","mi","si","sí","porque","esta","está","este","son","fue","muy","sin","sobre","también","hay","ni","les","nos","eso","esto","todo","cuando","donde","qué","cual","cuál"}


def _palabras(texto):
    vistas = []
    for palabra in str(texto).split():
        limpia = palabra.strip(".,;:¡!¿?()\"'").lower()
        if len(limpia) > 1 and limpia not in vistas and limpia not in PALABRAS_VACIAS:
            vistas.append(limpia)
    return vistas[:40]


def _quitar_palabra(texto, palabra):
    partes = []
    for p in str(texto).split():
        if p.strip(".,;:¡!¿?()\"'").lower() != palabra:
            partes.append(p)
    return " ".join(partes)


def explicar(modelo, meta, fila, cuantas=3):
    """Razones de una predicción: cuánto cambia la probabilidad si cada variable toma un valor de referencia
    (la mediana o la categoría más común del entrenamiento). En el texto, cuánto cambia al quitar cada palabra.
    Muestra de qué depende la predicción del modelo; no dice qué pasaría si el cliente cambiara esa variable."""
    base = fila[columnas_de_entrada(meta)].iloc[[0]].copy()
    p_base = float(predecir_probabilidad(modelo, meta, base)[0])
    variantes = []
    nombres = []
    for col in meta["columnas"]:
        nombre = col["nombre"]
        if col["tipo"] == "texto":
            for palabra in _palabras(base.iloc[0][nombre]):
                v = base.copy()
                v[nombre] = _quitar_palabra(base.iloc[0][nombre], palabra)
                variantes.append(v)
                nombres.append((nombre, f"«{palabra}»"))
        else:
            ref = col.get("referencia")
            if ref is None or str(base.iloc[0][nombre]) == str(ref):
                continue
            v = base.copy()
            v[nombre] = ref
            variantes.append(v)
            nombres.append((nombre, base.iloc[0][nombre]))
    if not variantes:
        return p_base, []
    otras = predecir_probabilidad(modelo, meta, pd.concat(variantes, ignore_index=True))
    razones = []
    for (nombre, valor), p in zip(nombres, otras):
        efecto = p_base - float(p)
        razones.append({"variable": nombre, "etiqueta": etiqueta(meta, nombre), "valor": valor,
                        "efecto": efecto, "sentido": "sube" if efecto > 0 else "baja"})
    razones = [r for r in razones if abs(r["efecto"]) >= 0.0005]
    razones.sort(key=lambda r: abs(r["efecto"]), reverse=True)
    return p_base, razones[:cuantas]


def _formato(valor):
    if isinstance(valor, (int, float, np.integer, np.floating)) and not isinstance(valor, bool):
        valor = float(valor)
        if valor.is_integer():
            return f"{valor:,.0f}".replace(",", ".")
        return f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return valor


def texto_razon(r, meta):
    """Una razón en lenguaje de negocio: 'Meses con importaciones = 2 sube la probabilidad 8 puntos'."""
    puntos = abs(r["efecto"]) * 100
    evento = meta.get("evento", "el evento")
    if str(r["valor"]).startswith("«"):
        return f"La palabra {r['valor']} {r['sentido']} la probabilidad de {evento} {puntos:.1f} puntos"
    return f"{r['etiqueta']} = {_formato(r['valor'])} {r['sentido']} la probabilidad de {evento} {puntos:.1f} puntos"


def evaluar(modelo, meta, datos, cuantas=3):
    """Probabilidad, decisión y razones de cada fila. Es la función que usa la interfaz."""
    salida = []
    for i in range(len(datos)):
        fila = datos.iloc[[i]]
        p, razones = explicar(modelo, meta, fila, cuantas)
        salida.append({"probabilidad": p, "decision": decidir(p, meta)[0], "razones": razones})
    return salida
