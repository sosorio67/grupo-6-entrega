"""ZONA FIJA: arma el paquete de datos para la opción HTML y lo inyecta en web/index.html.
No modifiquen este archivo: check.py verifica que siga igual.
Uso:  python -m core.web      (escribe web/dist/app.html, que se abre con doble clic)"""
import base64
import json
import math

from core.modelo import RAIZ, cargar_config, cargar_metadata, cargar_resultados

MARCA = "/*__PAQUETE__*/null"
LIMITE_FILAS = 3000   # la página HTML lleva a lo sumo las 3.000 filas de mayor probabilidad


def _limpio(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def paquete():
    """Todo lo que necesita la página: el caso, la regla de decisión, las métricas y los resultados."""
    config = cargar_config()
    meta = cargar_metadata()
    res = cargar_resultados()
    filas = []
    if res is not None:
        res = res.sort_values("probabilidad", ascending=False).head(LIMITE_FILAS)
        for registro in res.to_dict(orient="records"):
            filas.append({k: _limpio(v) for k, v in registro.items()})
    logo = RAIZ / (config["logo"] or "assets/logo.png")
    logo_b64 = base64.b64encode(logo.read_bytes()).decode() if logo.is_file() else None
    return {
        "app": config,
        "decision": {"umbrales": meta["umbrales"], "decisiones": meta["decisiones"], "evento": meta.get("evento", "")},
        "metricas": meta.get("metricas", {}),
        "etiquetas": meta.get("etiquetas", {}),
        "columna_id": meta.get("columna_id"),
        "resultados": filas,
        "logo": logo_b64,
    }


def construir_html():
    """Devuelve la página completa (un solo archivo) con los datos adentro."""
    plantilla = (RAIZ / "web" / "index.html").read_text(encoding="utf-8")
    if MARCA not in plantilla:
        raise ValueError("web/index.html debe conservar la línea  const P = /*__PAQUETE__*/null;")
    datos = json.dumps(paquete(), ensure_ascii=False).replace("</", "<\\/")
    return plantilla.replace(MARCA, datos)


def guardar_html():
    destino = RAIZ / "web" / "dist" / "app.html"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(construir_html(), encoding="utf-8")
    return destino


if __name__ == "__main__":
    print("Escrito:", guardar_html())
