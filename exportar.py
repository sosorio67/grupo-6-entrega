"""ZONA FIJA: exporta el modelo entrenado en Colab al formato que lee la plantilla.
No modifiquen este archivo: check.py verifica que siga igual.

Uso (en Colab o en su computador, después de entrenar):
    from exportar import exportar
    exportar(modelo=mi_pipeline, X_entrenamiento=X_train, umbrales=[0.35], decisiones=[...], metricas={...}, evento="...")

Escribe la carpeta model/ dentro de la carpeta de la app (donde está este archivo) y actualiza requirements.txt
con las versiones exactas de las librerías. Después sigan con desplegar.py."""
import datetime
import json
import re
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PALABRAS_VACIAS = {"de","la","que","el","en","y","a","los","se","del","las","un","por","con","no","una","su","para","es","al","lo","como","más","mas","pero","sus","le","ya","o","me","mi","si","sí","porque","esta","está","este","son","fue","muy","sin","sobre","también","hay","ni","les","nos","eso","esto","todo","cuando","donde","qué","cual","cuál"}

METRICAS_OBLIGATORIAS = ["costo_modelo", "costo_regla", "regla_simple", "unidad_costo"]


def _tipo(serie, nombre, columnas_texto):
    if nombre in columnas_texto:
        return "texto"
    if pd.api.types.is_bool_dtype(serie):
        return "categórica"
    if pd.api.types.is_numeric_dtype(serie):
        return "numérica"
    largo = serie.dropna().astype(str).str.len()
    if len(largo) and largo.mean() > 40:
        return "texto"
    return "categórica"


def _esquema(X, columnas_texto):
    columnas = []
    for nombre in X.columns:
        s = X[nombre]
        tipo = _tipo(s, nombre, columnas_texto)
        col = {"nombre": nombre, "tipo": tipo, "permite_faltantes": bool(s.isna().any())}
        if tipo == "numérica":
            n = pd.to_numeric(s, errors="coerce")
            col.update(minimo=float(n.min()), maximo=float(n.max()), referencia=float(n.median()),
                       p10=float(n.quantile(0.10)), p90=float(n.quantile(0.90)))
        elif tipo == "categórica":
            frecuencias = s.dropna().astype(str).value_counts()
            col.update(categorias=list(frecuencias.index[:300]),
                       referencia=frecuencias.index[0] if len(frecuencias) else None)
        columnas.append(col)
    return columnas


def _ejemplo_entrada(columnas):
    """Tres filas inventadas a partir de valores típicos (no son registros reales de los datos)."""
    filas = []
    for punto in ["referencia", "p10", "p90"]:
        fila = {}
        for c in columnas:
            if c["tipo"] == "numérica":
                fila[c["nombre"]] = c.get(punto, c["referencia"])
            elif c["tipo"] == "categórica":
                fila[c["nombre"]] = c.get("referencia")
            else:
                fila[c["nombre"]] = "Texto de ejemplo escrito para probar el modelo"
        filas.append(fila)
    return pd.DataFrame(filas)


def _versiones(modelo):
    import sklearn
    usados = {"scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
              "joblib": joblib.__version__}
    import pickle
    contenido = pickle.dumps(modelo)
    for paquete in ["xgboost", "lightgbm", "catboost"]:
        if paquete.encode() in contenido and paquete in sys.modules:
            usados[paquete] = sys.modules[paquete].__version__
    return usados


def _igual(a, b):
    if str(a) == str(b):
        return True
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return False


def _proba(modelo, X, clase_positiva):
    clases = list(modelo.classes_)
    for i, c in enumerate(clases):
        if _igual(c, clase_positiva):
            return modelo.predict_proba(X)[:, i]
    raise ValueError(f"La clase positiva {clase_positiva!r} no está entre las clases del modelo: {clases}")


def _decidir(p, umbrales, decisiones):
    return [decisiones[i] for i in np.searchsorted(np.array(umbrales), p, side="right")]


def _razones(modelo, X, columnas, clase_positiva, etiquetas, ocultas, cuantas=3):
    """Las mismas razones que muestra la app (core/modelo.py), calculadas para cada fila de la tabla."""
    textos = []
    reservadas = set(ocultas)
    X = X.reset_index(drop=True)
    for inicio in range(0, len(X), 300):
        bloque = X.iloc[inicio:inicio + 300].reset_index(drop=True)
        p0 = _proba(modelo, bloque, clase_positiva)
        variantes, quien = [], []
        for i in range(len(bloque)):
            base = bloque.iloc[[i]]
            for c in columnas:
                n = c["nombre"]
                if c["tipo"] == "texto":
                    if len(X) > 200:
                        continue
                    texto = str(base.iloc[0][n])
                    palabras = []
                    for w in texto.split():
                        w = w.strip(".,;:¡!¿?()\"'").lower()
                        if len(w) > 1 and w not in palabras and w not in PALABRAS_VACIAS:
                            palabras.append(w)
                    for w in palabras[:40]:
                        v = base.copy()
                        v[n] = " ".join(t for t in texto.split() if t.strip(".,;:¡!¿?()\"'").lower() != w)
                        variantes.append(v)
                        quien.append((i, n, f"«{w}»"))
                    continue
                ref = c.get("referencia")
                if ref is None or str(base.iloc[0][n]) == str(ref):
                    continue
                v = base.copy()
                v[n] = ref
                variantes.append(v)
                quien.append((i, n, base.iloc[0][n]))
        efectos = {i: [] for i in range(len(bloque))}
        if variantes:
            ps = _proba(modelo, pd.concat(variantes, ignore_index=True), clase_positiva)
            for (i, n, valor), p in zip(quien, ps):
                efectos[i].append((n, valor, p0[i] - p))
        for i in range(len(bloque)):
            partes = []
            utiles = [t for t in efectos[i] if abs(t[2]) >= 0.0005]
            for n, valor, e in sorted(utiles, key=lambda t: abs(t[2]), reverse=True)[:cuantas]:
                sentido = "sube" if e > 0 else "baja"
                if isinstance(valor, float):
                    valor = f"{valor:,.2f}".rstrip("0").rstrip(".").replace(",", "X").replace(".", ",").replace("X", ".")
                if str(valor).startswith("«"):
                    partes.append(f"{valor} {sentido} {abs(e) * 100:.1f} pts")
                elif n in reservadas:
                    partes.append(f"{etiquetas.get(n, n)} {sentido} {abs(e) * 100:.1f} pts")
                else:
                    partes.append(f"{etiquetas.get(n, n)} = {valor} {sentido} {abs(e) * 100:.1f} pts")
            textos.append(" · ".join(partes))
        if inicio and inicio % 3000 == 0:
            print(f"   razones: {inicio} de {len(X)} filas")
    return textos


def _rango(valor, cortes, prefijo):
    if pd.isna(valor):
        return None
    for a, b in zip(cortes[:-1], cortes[1:]):
        if a <= valor < b:
            if b == float("inf"):
                return f"más de {prefijo}{a:,.0f}".replace(",", ".")
            return f"{prefijo}{a:,.0f} a {prefijo}{b:,.0f}".replace(",", ".")
    return None


def _proteger(tabla, avisos, columna_id, columnas_ocultas, columnas_en_rangos, enmascarar_texto):
    """Aplica a la tabla de resultados las condiciones de uso de los datos que el equipo indicó."""
    tabla = tabla.copy()
    for col in columnas_ocultas:
        if col in tabla.columns and col != columna_id:
            tabla = tabla.drop(columns=col)
            avisos.append(f"Quité {col} de los resultados (está en columnas_ocultas).")
    for col, regla in columnas_en_rangos.items():
        if col in tabla.columns:
            cortes = [float(x) for x in regla["cortes"]] + [float("inf")]
            tabla[f"rango_{col}"] = [_rango(v, cortes, regla.get("prefijo", "")) for v in tabla[col]]
            tabla = tabla.drop(columns=col)
            avisos.append(f"Cambié {col} por rango_{col}.")
    if enmascarar_texto:
        for col in tabla.columns:
            if tabla[col].dtype == object and col != columna_id:
                tabla[col] = tabla[col].astype(str).str.replace(r"\S+@\S+", "[correo]", regex=True)
                tabla[col] = tabla[col].str.replace(r"\d[\d\s\-\.]{5,}\d", "[número]", regex=True)
    return tabla


def _actualizar_requisitos(ruta, versiones):
    """Deja en requirements.txt las versiones exactas con que se entrenó el modelo."""
    lineas = ruta.read_text(encoding="utf-8").splitlines() if ruta.exists() else ["streamlit"]
    nuevas = []
    for linea in lineas:
        nombre = re.split(r"[=<>~!\s#]", linea.strip(), maxsplit=1)[0].lower()
        if nombre in versiones or (linea.strip().startswith("#") and any(p in linea for p in versiones)):
            continue
        nuevas.append(linea)
    for paquete, version in versiones.items():
        nuevas.append(f"{paquete}=={version}")
    ruta.write_text("\n".join(nuevas) + "\n", encoding="utf-8")


def exportar(modelo, X_entrenamiento, umbrales, decisiones, metricas, evento, clase_positiva=1,
             tabla_resultados=None, probabilidad_resultados=None, columna_id=None, columnas_a_mostrar=None,
             columnas_texto=None, etiquetas=None, columnas_ocultas=None, columnas_en_rangos=None,
             max_filas_resultados=None, enmascarar_texto=False, carpeta=None):
    """Exporta el modelo y sus resultados a la carpeta model/.

    Obligatorios
    modelo                  pipeline de scikit-learn ya entrenado (preprocesamiento + modelo), con predict_proba
    X_entrenamiento         DataFrame con las columnas de entrada, tal como entran al pipeline
    umbrales                lista de umbrales en orden creciente, por ejemplo [0.35] o [0.20, 0.45]
    decisiones              una decisión más que umbrales, de menor a mayor probabilidad
    metricas                diccionario con costo_modelo, costo_regla, regla_simple y unidad_costo
    evento                  lo que predice el modelo, en palabras (por ejemplo, "abandono" o "atraso")

    Opcionales
    clase_positiva          valor del target que es el evento (por defecto 1)
    tabla_resultados        filas que la app mostrará como lista, radar o mapa (con las columnas de entrada)
    probabilidad_resultados probabilidades ya calculadas para esa tabla (por ejemplo, con validación cruzada)
    columna_id              columna que identifica cada fila de la tabla de resultados
    columnas_a_mostrar      otras columnas de esa tabla que la app puede mostrar
    columnas_texto          columnas de entrada que son texto libre
    etiquetas               nombres de negocio de las columnas, por ejemplo {"dist_tm_m": "Distancia a TransMilenio"}

    Condiciones de uso de los datos (revisen las de su caso en el brief)
    columnas_ocultas        columnas que la app no puede mostrar; tampoco aparece su valor en las razones
    columnas_en_rangos      columnas que solo se pueden mostrar en rangos: {"columna": {"cortes": [0, 100, 500], "prefijo": "US$"}}
    max_filas_resultados    número máximo de filas de la tabla de resultados
    enmascarar_texto        True para cambiar números largos y correos del texto por [número] y [correo]
    """
    columnas_ocultas = list(columnas_ocultas or [])
    columnas_en_rangos = dict(columnas_en_rangos or {})
    ocultas = set(columnas_ocultas) | set(columnas_en_rangos)
    avisos = []
    app = Path(__file__).resolve().parent
    carpeta = Path(carpeta) if carpeta else app / "model"
    carpeta.mkdir(parents=True, exist_ok=True)
    columnas_texto = list(columnas_texto or [])
    etiquetas = dict(etiquetas or {})
    if not hasattr(modelo, "predict_proba"):
        raise ValueError("El modelo debe tener predict_proba (un clasificador de scikit-learn o compatible).")
    if not isinstance(X_entrenamiento, pd.DataFrame):
        raise ValueError("X_entrenamiento debe ser un DataFrame con los nombres de las columnas.")
    umbrales = [float(u) for u in umbrales]
    if umbrales != sorted(umbrales) or any(u <= 0 or u >= 1 for u in umbrales):
        raise ValueError("Los umbrales deben estar entre 0 y 1 y en orden creciente.")
    if len(decisiones) != len(umbrales) + 1:
        raise ValueError(f"Con {len(umbrales)} umbral(es) se necesitan {len(umbrales) + 1} decisiones.")
    faltan = [m for m in METRICAS_OBLIGATORIAS if m not in metricas]
    if faltan:
        avisos.append(f"Faltan métricas obligatorias: {', '.join(faltan)}. check.py no dará LISTO sin ellas.")
    columnas = _esquema(X_entrenamiento, columnas_texto)
    _proba(modelo, X_entrenamiento.head(3), clase_positiva)

    joblib.dump(modelo, carpeta / "pipeline.joblib", compress=3)
    _ejemplo_entrada(columnas).to_csv(carpeta / "ejemplo_entrada.csv", index=False)

    if tabla_resultados is not None:
        tabla = tabla_resultados.reset_index(drop=True)
        if probabilidad_resultados is not None:
            tabla["_p"] = np.asarray(probabilidad_resultados, dtype=float)
        if max_filas_resultados and len(tabla) > max_filas_resultados:
            tabla = tabla.head(max_filas_resultados)
            avisos.append(f"Dejé solo {max_filas_resultados} filas en los resultados (max_filas_resultados).")
        X = tabla[[c["nombre"] for c in columnas]]
        if probabilidad_resultados is not None:
            p = tabla["_p"].to_numpy()
        else:
            p = _proba(modelo, X, clase_positiva)
        mostrar = []
        for c in [columna_id] + list(columnas_a_mostrar or []):
            if c and c in tabla.columns and c not in mostrar:
                mostrar.append(c)
        salida = tabla[mostrar].copy()
        if columna_id in salida.columns:
            salida[columna_id] = [str(int(v)) if isinstance(v, float) and v.is_integer() else v
                                  for v in salida[columna_id]]
        salida["probabilidad"] = np.round(p, 4)
        salida["decision"] = _decidir(p, umbrales, decisiones)
        print(f"Calculando las razones de {len(X)} filas...")
        salida["razones"] = _razones(modelo, X, columnas, clase_positiva, etiquetas, ocultas)
        salida = _proteger(salida, avisos, columna_id, columnas_ocultas, columnas_en_rangos, enmascarar_texto)
        salida.to_csv(carpeta / "resultados.csv", index=False)

    meta = {
        "evento": evento,
        "clase_positiva": clase_positiva, "umbrales": umbrales, "decisiones": list(decisiones),
        "metricas": metricas, "columnas": columnas, "etiquetas": etiquetas, "columna_id": columna_id,
        "n_entrenamiento": int(len(X_entrenamiento)), "versiones": _versiones(modelo),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "fecha_exportacion": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(carpeta / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

    if carpeta.parent == app:
        _actualizar_requisitos(app / "requirements.txt", meta["versiones"])
        print(f"\nListo: {carpeta} y requirements.txt actualizados con estas versiones:")
    else:
        print(f"\nListo: {carpeta}. Copien esa carpeta a la app y pongan en requirements.txt estas versiones:")
    for paquete, version in meta["versiones"].items():
        print(f"   {paquete}=={version}")
    print(f"Modelo exportado con Python {meta['python']}: elijan esa versión al publicar en Streamlit.")
    for a in avisos:
        print("AVISO:", a)
    en_colab = "google.colab" in sys.modules
    if en_colab and not str(carpeta).startswith("/content/drive"):
        zip_ruta = shutil.make_archive(str(carpeta), "zip", root_dir=carpeta.parent, base_dir=carpeta.name)
        try:
            from google.colab import files
            files.download(zip_ruta)
        except Exception:
            pass
    print("Siguiente paso: desplegar (python desplegar.py, o desplegar(...) en Colab).")
    return meta
