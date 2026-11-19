"""ZONA FIJA de la plantilla: no se modifica. La interfaz (zona libre) está en app.py, mi_app.py y web/."""
from core.modelo import (cargar_config, cargar_metadata, cargar_modelo, cargar_resultados, columnas_de_entrada,
                         decidir, etiqueta, evaluar, explicar, hay_modelo, predecir_probabilidad, texto_razon)
from core.validar import validar
