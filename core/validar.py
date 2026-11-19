"""ZONA FIJA: revisa que los datos de entrada tengan el formato con que se entrenó el modelo.
No modifiquen este archivo: check.py verifica que siga igual."""
import pandas as pd


def validar(datos, meta):
    """Devuelve (datos_listos, errores, avisos).
    Un error impide predecir (falta una columna, un número no es número).
    Un aviso deja predecir pero advierte (un valor fuera del rango visto al entrenar, una categoría nueva)."""
    errores = []
    avisos = []
    datos = datos.copy()
    for col in meta["columnas"]:
        nombre = col["nombre"]
        if nombre not in datos.columns:
            errores.append(f"Falta la columna {nombre}.")
            continue
        if col["tipo"] == "numérica":
            convertida = pd.to_numeric(datos[nombre], errors="coerce")
            malos = convertida.isna() & datos[nombre].notna() & (datos[nombre].astype(str).str.strip() != "")
            if malos.any():
                errores.append(f"{nombre} debe ser un número.")
            datos[nombre] = convertida
            fuera = (convertida < col["minimo"]) | (convertida > col["maximo"])
            if fuera.any():
                avisos.append(f"{nombre} está fuera del rango visto al entrenar "
                              f"({col['minimo']:g} a {col['maximo']:g}); la predicción es menos confiable.")
            if convertida.isna().any() and not col.get("permite_faltantes", False):
                avisos.append(f"{nombre} viene vacía y en el entrenamiento nunca lo estuvo.")
        elif col["tipo"] == "categórica":
            datos[nombre] = datos[nombre].astype(object).where(datos[nombre].notna(), None)
            conocidas = set(str(c) for c in col.get("categorias", []))
            nuevas = set(str(v) for v in datos[nombre].dropna()) - conocidas
            if conocidas and nuevas:
                avisos.append(f"{nombre} tiene valores que el modelo no vio al entrenar: {', '.join(sorted(nuevas))[:120]}.")
        elif col["tipo"] == "texto":
            datos[nombre] = datos[nombre].fillna("").astype(str)
            if (datos[nombre].str.strip() == "").any():
                avisos.append(f"{nombre} viene vacío.")
    return datos, errores, avisos
