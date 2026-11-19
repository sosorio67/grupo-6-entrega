"""ZONA LIBRE: la interfaz de la aplicación.
Ya funciona con cualquier modelo exportado con exportar.py. Desde aquí la adaptan a su caso:
cambien textos, orden, gráficos y colores, con ayuda de IA si quieren (PASO 5 de la guía).
Lo único que debe quedar es el pie de página al final.

Para verla en su computador:   streamlit run app.py"""
import pandas as pd
import streamlit as st

from core import (cargar_config, cargar_metadata, cargar_modelo, cargar_resultados, etiqueta, evaluar,
                  hay_modelo, texto_razon, validar)

config = cargar_config()
st.set_page_config(page_title=config["titulo"], layout="wide")


# ---------------------------------------------------------------------------------------------
# Encabezado: título, cliente y descripción vienen de mi_app.py
# ---------------------------------------------------------------------------------------------
st.title(config["titulo"])
st.caption("Cimiento Seguros · Suscripción y seguimiento")
# ---------------------------------------------------------------------------------------------
# Mientras no exporten su modelo, la app muestra la guía en lugar de fallar
# ---------------------------------------------------------------------------------------------
if not hay_modelo():
    st.info("Todavía no hay modelo en la carpeta model/. Sigan la guía del README:")
    st.markdown("""
1. Entrenen su modelo como un `Pipeline` de scikit-learn.
2. Expórtenlo con `exportar` (en Colab, cuaderno `Exportar_modelo_desde_Colab.ipynb`).
3. Completen `mi_app.py`.
4. Corran `desplegar` (en Colab, `Desplegar_desde_Colab.ipynb`; en su computador, `python desplegar.py`).
5. Adapten esta interfaz a su caso.
""")
    st.divider()
    st.caption(config["pie"])
    st.stop()


@st.cache_resource
def recursos():
    return cargar_metadata(), cargar_modelo(), cargar_resultados()


meta, modelo, resultados = recursos()
evento = meta["evento"]
unidad = config["unidad"]


def mostrar_resultado(probabilidad, decision, razones):
    """Cómo se ve el resultado de una evaluación. Cambien el formato si quieren."""
    c1, c2 = st.columns(2)
    c1.metric(f"Probabilidad de {evento}", f"{probabilidad:.0%}")
    c2.metric("Recomendación", decision)
    st.write(f"Índice de templanza contractual: {100*(1-probabilidad):.1f} sobre 100")
    st.caption("Estimación de prórroga. La recomendación requiere análisis del suscriptor y no acredita incumplimiento.")
    limites = modelo.parametros["cortes_nivel"]
    nivel = "Reforzada" if probabilidad >= limites[1] else "Estándar" if probabilidad >= limites[0] else "Ligera"
    st.write("Vigilancia sugerida: " + nivel)
    if not razones:
        st.caption("Con estos valores ninguna variable mueve la probabilidad frente a un caso típico.")
    else:
        st.markdown("**Razones principales**")
        for r in razones:
            if isinstance(r, dict):
                st.markdown(f"- {texto_razon(r, meta)}")
            else:
                st.markdown(f"- {r}")
        st.caption("Las razones muestran de qué depende la predicción del modelo, no qué pasaría si algo cambiara.")


def formulario_nuevo():
    """Un campo por cada variable de entrada del modelo, con valores típicos por defecto."""
    st.caption("Consulte los antecedentes disponibles al firmar. Cuando no haya contratos previos, ingrese cero en el conteo y la tasa de esa parte.")
    valores = {}
    with st.form("nuevo"):
        columnas = st.columns(2)
        for i, col in enumerate(meta["columnas"]):
            nombre = col["nombre"]
            texto = etiqueta(meta, nombre)
            with columnas[i % 2]:
                if nombre in ("fecha_de_firma", "fecha_de_inicio", "fin_pactado"):
                    fecha = st.date_input(texto, value=None, key=nombre)
                    valores[nombre] = fecha.isoformat() if fecha else None
                elif nombre in ("en_rues","puede_prorrogarse","es_consorcio_ut"):
                    valores[nombre] = st.selectbox(texto, [0,1], format_func=lambda x: "Sí" if x else "No", key=nombre)
                elif col["tipo"] == "numérica":
                    valores[nombre] = st.number_input(texto, value=float(col["referencia"]), key=nombre)
                elif col["tipo"] == "categórica":
                    valores[nombre] = st.selectbox(texto, col.get("categorias") or [""], key=nombre)
                else:
                    valores[nombre] = st.text_area(texto, key=nombre)
        enviado = st.form_submit_button("Evaluar")
    if enviado:
        if any(not 0 <= valores[c] <= 1 for c in ("ent_tasa_atraso_prev","prov_tasa_atraso_prev")) or not valores["fecha_de_firma"] or min(valores["valor_del_contrato"],valores["duracion_pactada_dias"]) <= 0 or any(valores[c]<0 or not float(valores[c]).is_integer() for c in ("prov_n_previos_obs","ent_n_previos_obs")):
            st.error("Revise las tasas (0 a 1), la fecha de firma, el valor, el plazo y el número de antecedentes.")
            return
        import unicodedata
        valores["objeto_n"] = "".join(c for c in unicodedata.normalize("NFKD",valores["objeto_n"].lower()) if not unicodedata.combining(c))
        datos, errores, avisos = validar(pd.DataFrame([valores]), meta)
        for e in errores:
            st.error(e)
        for a in avisos:
            if not a.startswith(("fecha_de_firma ", "fecha_de_inicio ", "fin_pactado ")):
                st.warning(a)
        if not errores:
            r = evaluar(modelo, meta, datos)[0]
            mostrar_resultado(r["probabilidad"], r["decision"], r["razones"])


def buscar_en_lista():
    col_id = meta.get("columna_id")
    if resultados is None or not col_id or col_id not in resultados.columns:
        st.info("No exportaron una tabla de resultados con columna_id para buscar.")
        return
    elegido = st.selectbox(f"Busquen {unidad} por {etiqueta(meta, col_id)}", resultados[col_id].astype(str))
    fila = resultados[resultados[col_id].astype(str) == elegido].iloc[0]
    razones = [r for r in str(fila.get("razones", "")).split(" · ") if r and r != "nan"]
    mostrar_resultado(float(fila["probabilidad"]), fila["decision"], razones)


def vista_lista():
    if resultados is None:
        st.info("Exporten una tabla de resultados (tabla_resultados en exportar) para ver aquí la lista.")
        return
    tabla = resultados[resultados.grupo.eq("Radar")].sort_values("prioridad")
    if st.checkbox("Mostrar solo los 1.000 priorizados",value=True):
        tabla = tabla[tabla.prioridad <= 1000]
    for campo,et in [("departamento","Departamento"),("tipo_de_contrato","Tipo de contrato"),("nombre_entidad","Entidad")]:
        seleccion = st.multiselect(et, sorted(tabla[campo].dropna().unique()))
        if seleccion:
            tabla = tabla[tabla[campo].isin(seleccion)]
    st.caption(f"{len(tabla):,} contratos · La prioridad corresponde al radar completo.")
    st.download_button("Descargar selección",tabla.to_csv(index=False),"radar.csv","text/csv")
    st.dataframe(tabla, width="stretch", hide_index=True)


def vista_valor():
    st.caption("Prueba histórica 2024–2025. Para el radar y las consultas se usa el reajuste final del cuaderno con 47.238 observables.")
    m = meta["metricas"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Costo con el modelo", f"{m['costo_modelo']:.1f}".replace(".", ","))
    c2.metric("Costo con la regla simple", f"{m['costo_regla']:.1f}".replace(".", ","))
    c3.metric("Ahorro frente a la regla", f"{m['costo_regla'] - m['costo_modelo']:.1f}".replace(".", ","))
    st.caption(f"Costos {m['unidad_costo']}, en los datos de prueba.")
    st.markdown(f"**Regla simple:** {m['regla_simple']}")
    if "intervalo_diferencia" in m:
        a, b = m["intervalo_diferencia"]
        st.markdown(f"Intervalo de 95 % del ahorro: de {a:,.0f} a {b:.1f}".replace(".", ","))
    otras = {k: v for k, v in m.items()
             if k not in ("costo_modelo", "costo_regla", "regla_simple", "unidad_costo", "intervalo_diferencia")}
    if otras:
        st.table(pd.DataFrame({"Métrica": list(otras.keys()), "Valor": [str(v) for v in otras.values()]}))


def vista_regla():
    filas = []
    limites = [0.0] + meta["umbrales"] + [1.0]
    for i, d in enumerate(meta["decisiones"]):
        filas.append({f"Probabilidad de {evento}": f"{limites[i]:.2%} a {limites[i + 1]:.2%}", "Recomendación": d})
    st.table(pd.DataFrame(filas))
    st.caption("El cuaderno activa la alerta al 18 %. Como política propuesta para la interfaz, el nivel de vigilancia reforzada activa la revisión humana de garantías. No se calcula el precio de la prima.")
    st.caption(f"Modelo entrenado con {meta['n_entrenamiento']} filas; exportado el {meta['fecha_exportacion']}.")


# ---------------------------------------------------------------------------------------------
# Cuerpo de la app. PASO 5: adapten pestañas, textos y gráficos a lo que necesita su cliente.
# ---------------------------------------------------------------------------------------------
if config["interfaz"] == "html":
    from core.web import construir_html
    st.iframe(construir_html(), height=820)
    with st.expander(f"Evaluar {unidad} nuevo"):
        formulario_nuevo()
else:
    pestanas = st.tabs([f"Evaluar {unidad}", "Radar de vigilancia", "Cuánto vale el modelo", "Cómo decide"])
    with pestanas[0]:
        modo = st.radio("¿Qué quieren hacer?", ["Escribir los datos", "Buscar en la lista"], horizontal=True)
        if modo == "Escribir los datos":
            formulario_nuevo()
        else:
            buscar_en_lista()
    with pestanas[1]:
        vista_lista()
    with pestanas[2]:
        vista_valor()
    with pestanas[3]:
        vista_regla()


# Pie de página: debe quedar (atribuciones y advertencias de su caso, en mi_app.py)
st.divider()
st.caption(config["pie"])
