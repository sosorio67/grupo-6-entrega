"""Revisión antes de publicar. Córranlo desde la carpeta de la app:

    python check.py

Debe terminar en LISTO PARA DESPLEGAR. Si algo falla, el mensaje dice qué arreglar.
No modifiquen este archivo."""
import hashlib
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
HUELLAS = {
    "core/__init__.py": "def2bad31737d64e5f32156374e44dc7510581dc5ff68e0fe6010f68897247ae",
    "core/modelo.py": "545579b7030524a19c1e0e2ac7549952de352dabc422adfd226a0115d3b6da2f",
    "core/validar.py": "24189bbfd95b6f5c634436a80b439c5a0fcd3be3a79b131d98bf751abe4758bf",
    "core/web.py": "72390cb2618b403864b06e4ddc76e69f8a73e8d532ac10bf5f980cb9a95955bd",
    "exportar.py": "bdbe129aa935f6474b109fa121de18f8d205da95920f14c6dd8baa512b4ca59b",
    "desplegar.py": "acf6f3080bd836e6e2fd5edcb0b1cbd11aff8c46ae6924f862c832d897bdd09a"
}
DATOS_DEL_CURSO = {"glowup_data.csv", "glowup_tabla_analitica_uso_interno.csv", "glowup_tabla_abierta.csv",
                   "cafescore_data.csv", "cafescore_hexes.csv", "appodio_resenas.csv",
                   "appodio_muestra_para_etiquetar.csv", "appodio_panel_app_mes.csv", "obrasegura_contratos.csv",
                   "obrasegura_data.csv"}

fallas = []


def paso(nombre, funcion):
    try:
        detalle = funcion()
        print(f"[OK]    {nombre}" + (f": {detalle}" if detalle else ""))
        return True
    except Exception as e:
        fallas.append(nombre)
        print(f"[FALLA] {nombre}: {e}")
        return False


def archivos_de_la_plantilla():
    faltan = [f for f in ["app.py", "mi_app.py", "requirements.txt", "exportar.py", "desplegar.py", "core/__init__.py",
                          "core/modelo.py", "core/validar.py", "core/web.py", "web/index.html"]
              if not (RAIZ / f).exists()]
    if faltan:
        raise RuntimeError("faltan " + ", ".join(faltan))


def zona_fija_intacta():
    cambiados = []
    for ruta, huella in HUELLAS.items():
        contenido = (RAIZ / ruta).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(contenido).hexdigest() != huella:
            cambiados.append(ruta)
    if cambiados:
        raise RuntimeError("se modificaron archivos de la zona fija: " + ", ".join(cambiados) +
                           ". Vuelvan a copiarlos de la plantilla original.")


def mi_app_completo():
    texto = (RAIZ / "mi_app.py").read_text(encoding="utf-8")
    pendientes = [linea.split("=")[0].strip() for linea in texto.splitlines()
                  if "COMPLETAR" in linea and "=" in linea and not linea.strip().startswith("#")]
    if pendientes:
        raise RuntimeError("completen en mi_app.py: " + ", ".join(pendientes))


def modelo_exportado():
    from core import cargar_metadata
    faltan = [f for f in ["pipeline.joblib", "metadata.json", "ejemplo_entrada.csv"] if not (RAIZ / "model" / f).exists()]
    if faltan:
        raise RuntimeError("falta exportar el modelo (model/" + ", model/".join(faltan) + "). Usen exportar (PASO 3 del README)")
    meta = cargar_metadata()
    faltan = [m for m in ["costo_modelo", "costo_regla", "regla_simple", "unidad_costo"] if m not in meta.get("metricas", {})]
    if faltan:
        raise RuntimeError("faltan métricas en exportar(metricas=...): " + ", ".join(faltan))
    return f"{len(meta['columnas'])} variables, umbrales {meta['umbrales']}, decisiones {meta['decisiones']}"


def versiones():
    from core import cargar_metadata
    meta = cargar_metadata()
    fijadas = {}
    for linea in (RAIZ / "requirements.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*([A-Za-z0-9_\-]+)\s*==\s*([^\s;#]+)", linea)
        if m:
            fijadas[m.group(1).lower()] = m.group(2)
    problemas = []
    for paquete, version in meta.get("versiones", {}).items():
        if paquete in ("scikit-learn", "xgboost", "lightgbm", "catboost") and fijadas.get(paquete) != version:
            problemas.append(f"{paquete}=={version}")
    if "streamlit" not in fijadas:
        problemas.append("streamlit==<versión>")
    if problemas:
        raise RuntimeError("en requirements.txt deben quedar exactamente: " + ", ".join(problemas))
    import sklearn
    esperada = meta.get("versiones", {}).get("scikit-learn")
    if esperada and sklearn.__version__ != esperada:
        raise RuntimeError(f"tienen instalado scikit-learn {sklearn.__version__} y el modelo se entrenó con {esperada}: "
                           f"corran  pip install -r requirements.txt")


def modelo_predice():
    import pandas as pd
    from core import cargar_metadata, cargar_modelo, evaluar, validar
    meta, modelo = cargar_metadata(), cargar_modelo()
    datos, errores, _ = validar(pd.read_csv(RAIZ / "model" / "ejemplo_entrada.csv"), meta)
    if errores:
        raise RuntimeError("; ".join(errores))
    salida = evaluar(modelo, meta, datos)
    for r in salida:
        if not 0 <= r["probabilidad"] <= 1:
            raise RuntimeError("una probabilidad está fuera de 0 a 1")
        if r["decision"] not in meta["decisiones"]:
            raise RuntimeError("una decisión no está en la lista de decisiones")
    return ", ".join(f"{r['probabilidad']:.0%} → {r['decision']}" for r in salida)


def resultados_validos():
    from core import cargar_resultados
    res = cargar_resultados()
    if res is None:
        return "no hay model/resultados.csv (es opcional)"
    for c in ["probabilidad", "decision"]:
        if c not in res.columns:
            raise RuntimeError(f"resultados.csv no tiene la columna {c}; expórtenlo con exportar.py")
    return f"{len(res)} filas. Revisen que cumpla las condiciones de uso de los datos de su caso"


def sin_datos_del_curso():
    problemas = []
    for f in RAIZ.rglob("*"):
        if not f.is_file() or ".git" in f.parts or ".venv" in f.parts or "venv" in f.parts:
            continue
        if f.name.lower() in DATOS_DEL_CURSO:
            problemas.append(f"{f.relative_to(RAIZ)} (datos del curso)")
        elif f.stat().st_size > 95e6:
            problemas.append(f"{f.relative_to(RAIZ)} pesa más de 95 MB (GitHub no lo acepta)")
        elif f.suffix.lower() in (".csv", ".parquet", ".xlsx") and f.parent.name != "model" and f.stat().st_size > 1e6:
            problemas.append(f"{f.relative_to(RAIZ)} parece una tabla de datos")
    if problemas:
        raise RuntimeError("el repositorio no puede llevar los datos del curso: " + "; ".join(problemas))


def pie_presente():
    if 'config["pie"]' not in (RAIZ / "app.py").read_text(encoding="utf-8"):
        raise RuntimeError('app.py debe mostrar el pie de página (config["pie"]) al final')
    if "/*__PAQUETE__*/null" not in (RAIZ / "web" / "index.html").read_text(encoding="utf-8"):
        raise RuntimeError("web/index.html debe conservar  const P = /*__PAQUETE__*/null;")


def html_se_arma():
    from core.web import construir_html
    return f"{len(construir_html()) / 1e6:.1f} MB"


def app_corre():
    import logging
    logging.getLogger("streamlit").setLevel(logging.ERROR)
    from streamlit.testing.v1 import AppTest
    prueba = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=180).run()
    if prueba.exception:
        raise RuntimeError(prueba.exception[0].value)


print("Revisando la app...\n")
paso("archivos de la plantilla", archivos_de_la_plantilla)
paso("zona fija sin cambios", zona_fija_intacta)
paso("mi_app.py completo", mi_app_completo)
if paso("modelo exportado con sus métricas", modelo_exportado):
    paso("versiones en requirements.txt", versiones)
    paso("el modelo carga, predice y explica", modelo_predice)
    paso("resultados.csv", resultados_validos)
    paso("la opción HTML se arma", html_se_arma)
paso("el repositorio no lleva los datos del curso", sin_datos_del_curso)
paso("pie de página y marca de datos del HTML", pie_presente)
paso("app.py corre de principio a fin", app_corre)

if fallas:
    print(f"\nHAY {len(fallas)} PROBLEMA(S): no publiquen todavía.")
    sys.exit(1)
print("\nLISTO PARA DESPLEGAR")
