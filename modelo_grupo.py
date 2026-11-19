import numpy as np
import pandas as pd

numericas = ["valor_log", "valor_por_dia_log", "duracion_pactada_dias", "plazo_relativo", "edad_proveedor_anios",
             "exp_prov_log", "exp_ent_log", "prov_tasa_suav", "ent_tasa_suav", "dias_firma_inicio", "pct_anticipo"]
indicadoras = ["prov_sin_historial", "ent_sin_historial", "firma_q4", "cierra_en_dic", "anticipo",
               "puede_prorrogarse", "es_consorcio_ut", "en_rues"]
categoricas = ["tipo_de_contrato", "modalidad_de_contratacion", "orden", "sector", "es_pyme",
               "origen_de_los_recursos", "destino_gasto", "tipo_obra"]
PESO_SUAVIZADO = 5     # "contratos imaginarios" con la tasa base que se suman al historial


def crear_variables(tabla, P):
    t = tabla.copy()
    texto = t["objeto_n"].fillna("")

    # Qué se está construyendo (leído del texto del objeto del contrato)
    t["tipo_obra"] = np.select(
        [texto.str.contains(r"\b(?:via|vias|vial|carretera|pavimento|puente)\b"),
         texto.str.contains(r"acueducto|alcantarillado|agua potable"),
         texto.str.contains(r"colegio|escuela|hospital|vivienda|edificio|sede|parque|coliseo"),
         texto.str.contains(r"alumbrado|electric|energia")],
        ["vias", "agua", "edificaciones", "energia"], default="otro")

    # Tamaño e intensidad
    t["valor_log"] = np.log1p(t["valor_del_contrato"])
    t["valor_por_dia_log"] = np.log1p(t["valor_del_contrato"] / t["duracion_pactada_dias"])
    t["plazo_relativo"] = t["duracion_pactada_dias"] / t["tipo_de_contrato"].map(P["mediana_plazo"])

    # Experiencia e historial (suavizado hacia la tasa base)
    t["prov_sin_historial"] = (t["prov_n_previos_obs"] == 0).astype(int)
    t["ent_sin_historial"] = (t["ent_n_previos_obs"] == 0).astype(int)
    t["exp_prov_log"] = np.log1p(t["prov_n_firmados_prev"])
    t["exp_ent_log"] = np.log1p(t["ent_n_firmados_prev"])
    for quien, n_prev, tasa in [("prov", "prov_n_previos_obs", "prov_tasa_atraso_prev"),
                                ("ent", "ent_n_previos_obs", "ent_tasa_atraso_prev")]:
        tasa_obs = t[tasa].fillna(P["tasa_base"])
        t[f"{quien}_tasa_suav"] = (tasa_obs * t[n_prev] + PESO_SUAVIZADO * P["tasa_base"]) / (t[n_prev] + PESO_SUAVIZADO)

    # Presión de cierre y otras señales administrativas
    t["firma_q4"] = (t["mes_firma"] >= 10).astype(int)
    t["cierra_en_dic"] = t["fin_pactado"].dt.month.isin([11, 12]).astype(int)
    t["dias_firma_inicio"] = (t["fecha_de_inicio"] - t["fecha_de_firma"]).dt.days
    t["anticipo"] = (t["pct_anticipo"] > 0).astype(int)
    return t


def preparar(tabla, P):
    t = crear_variables(tabla, P)
    for c in numericas:
        t[c] = t[c].fillna(P["medianas"][c]).clip(P["corte_bajo"][c], P["corte_alto"][c])   # nulos y extremos
    t[categoricas] = t[categoricas].astype(str)
    X = pd.get_dummies(t[numericas + indicadoras + categoricas], columns=categoricas, dtype=float)
    if P["columnas"] is not None:
        X = X.reindex(columns=P["columnas"], fill_value=0.0)      # mismas columnas que en el entrenamiento
    return X



class ModeloGrupo:
    """Adaptador para las fechas y el modelo final ya entrenado en el cuaderno."""
    def __init__(self, bosque, parametros):
        self.bosque = bosque
        self.parametros = parametros
        self.classes_ = bosque.classes_

    def predict_proba(self, entradas):
        datos = entradas.copy()
        for fecha in ["fecha_de_firma", "fecha_de_inicio", "fin_pactado"]:
            datos[fecha] = pd.to_datetime(datos[fecha], errors="coerce")
        datos["mes_firma"] = datos["fecha_de_firma"].dt.month
        return self.bosque.predict_proba(preparar(datos, self.parametros))
