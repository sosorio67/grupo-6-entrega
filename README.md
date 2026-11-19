# Plantilla para la aplicación del proyecto final

Esta carpeta es el punto de partida de la aplicación. No trae ningún modelo: ustedes la van completando con el suyo siguiendo los pasos de abajo. Sirve para los cuatro casos.

```
plantilla_app/
├─ README.md                  esta guía
├─ mi_app.py                  PASO 4: los datos de su app (título, cliente, pie de página)
├─ app.py                     ZONA LIBRE: la interfaz en Streamlit; la adaptan a su caso
├─ web/index.html             ZONA LIBRE: la interfaz en HTML, si prefieren esa opción
├─ assets/                    su logo o imágenes
├─ core/                      ZONA FIJA: carga el modelo, valida los datos, predice, decide y explica
├─ exportar.py                ZONA FIJA: exporta su modelo a model/ y actualiza requirements.txt
├─ desplegar.py               ZONA FIJA: revisa, abre la app para probarla y prepara la publicación
├─ check.py                   revisión antes de publicar; debe terminar en LISTO PARA DESPLEGAR
├─ requirements.txt           librerías y versiones que instala Streamlit Community Cloud
├─ Exportar_modelo_desde_Colab.ipynb    las celdas para exportar al final de su cuaderno
└─ Desplegar_desde_Colab.ipynb          las celdas para desplegar desde Colab
```

La zona fija no se modifica: `check.py` lo verifica. Todo lo demás es suyo.

## PASO 1. Tengan la plantilla donde van a trabajar

- **Si trabajan en Colab:** descompriman la plantilla y suban la carpeta `plantilla_app` a su Google Drive. Ahí van a quedar el modelo y la app.
- **Si trabajan en su computador:** descompriman la plantilla en una carpeta de su computador.

En los dos casos, el repositorio de GitHub (privado y compartido con el profesor) lo crean en el PASO 5. Nunca lleva los archivos de datos del curso.

## PASO 2. Entrenen el modelo como un pipeline

La app le pasa al modelo los datos tal como vienen en la tabla, así que todo el preprocesamiento (imputar, escalar, codificar categorías, convertir texto en números) debe ir dentro de un `Pipeline` de scikit-learn. Por ejemplo:

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numericas = ["columna_1", "columna_2"]
categoricas = ["columna_3"]

preparacion = ColumnTransformer([
    ("num", Pipeline([("imputar", SimpleImputer(strategy="median")), ("escalar", StandardScaler())]), numericas),
    ("cat", Pipeline([("imputar", SimpleImputer(strategy="most_frequent")),
                      ("codificar", OneHotEncoder(handle_unknown="ignore"))]), categoricas),
])
mi_pipeline = Pipeline([("preparacion", preparacion), ("modelo", LogisticRegression(max_iter=1000))])
mi_pipeline.fit(X_train, y_train)
```

Para texto, usen `TfidfVectorizer` dentro del `ColumnTransformer`. El último paso puede ser cualquier clasificador con `predict_proba` (árbol, bosque aleatorio, XGBoost). Si construyen variables nuevas, constrúyanlas como columnas de `X_train` antes del pipeline.

## PASO 3. Exporten el modelo

Al final de su cuaderno usen `exportar` (en Colab, copien las celdas de `Exportar_modelo_desde_Colab.ipynb`; en su computador, importen `exportar.py` desde la carpeta de la app). `exportar` escribe la carpeta `model/` dentro de la carpeta de la app, pone en `requirements.txt` las versiones exactas de las librerías con que entrenaron e imprime la versión de Python que deben elegir al publicar.

| Argumento | Qué es |
|---|---|
| `modelo` | el pipeline entrenado |
| `X_entrenamiento` | el DataFrame con que lo entrenaron (solo columnas de entrada) |
| `umbrales` | sus umbrales, de menor a mayor: `[0.35]` o `[0.20, 0.45]` |
| `decisiones` | una decisión más que umbrales, de menor a mayor probabilidad |
| `metricas` | `costo_modelo`, `costo_regla`, `regla_simple`, `unidad_costo` (en la prueba); opcional `intervalo_diferencia` y otras |
| `evento` | lo que predice el modelo, en palabras: "abandono", "atraso"… |
| `tabla_resultados` | opcional: las filas que la app muestra como lista, radar o mapa, con las columnas de entrada. Si trae `lat` y `lon`, la app dibuja un mapa |
| `probabilidad_resultados` | opcional: probabilidades ya calculadas para esa tabla (por ejemplo, con validación cruzada) |
| `columna_id`, `columnas_a_mostrar` | qué columnas de esa tabla muestra la app |
| `columnas_texto`, `etiquetas` | columnas de texto libre y nombres de negocio de las columnas |

**Condiciones de uso de los datos.** El brief de su caso dice qué no se puede mostrar. `exportar` tiene cuatro opciones para cumplirlo en la tabla de resultados:

| Opción | Para qué |
|---|---|
| `columnas_ocultas` | columnas que la app no puede mostrar; tampoco aparece su valor en las razones |
| `columnas_en_rangos` | columnas que solo se muestran en rangos: `{"columna": {"cortes": [0, 100000, 500000], "prefijo": "US$"}}` |
| `max_filas_resultados` | número máximo de filas |
| `enmascarar_texto` | `True` cambia números largos y correos del texto por [número] y [correo] |

Si hay filas que deben quedar anónimas (por ejemplo, personas naturales), cambien esos valores en la tabla antes de exportarla.

## PASO 4. Completen mi_app.py

Cambien cada `COMPLETAR`: título de la herramienta, cliente, qué decide el cliente, qué evalúa la app y el pie de página con las atribuciones y advertencias que pide su caso.

## PASO 5. Prueben, adapten y publiquen con desplegar

`desplegar` detecta si están en Google Colab o en su computador y sigue el camino que corresponde:

- **En Colab:** abran `Desplegar_desde_Colab.ipynb`, cambien la ruta de la carpeta y corran las celdas.
- **En su computador:** en la terminal, desde la carpeta de la app, corran `python desplegar.py`.

Hace cinco cosas:

1. Instala las librerías de `requirements.txt`.
2. Corre `check.py`. Si algo falla, dice qué arreglar y se detiene.
3. Abre la app para que la prueben: en Colab da un enlace; en su computador se abre en el navegador.
4. Deja los archivos listos para GitHub, sin los datos del curso: en Colab descarga `para_subir_a_github.zip`; en su computador, si la carpeta ya es un repositorio de git, ofrece subir los cambios con un solo "s".
5. Explica, clic por clic, cómo publicarla en Streamlit Community Cloud y qué versión de Python elegir.

La app ya funciona con su modelo: evalúa un caso nuevo, busca en la lista, muestra cuánto vale el modelo frente a la regla y cómo decide. Desde ahí la adaptan a lo que necesita su cliente en `app.py`, con ayuda de IA si quieren, y vuelven a correr `desplegar` después de cada cambio grande.

**Opción HTML.** En `mi_app.py` pongan `INTERFAZ = "html"` y editen `web/index.html`, conservando la línea `const P = /*__PAQUETE__*/null;`, donde se inyectan los datos. Para verla sin Streamlit: `python -m core.web` y abran `web/dist/app.html`. En esta opción, `app.py` agrega debajo un formulario para evaluar casos nuevos con el modelo.

## PASO 6. Publiquen

`desplegar` les muestra los pasos. En resumen: suban los archivos a su repositorio privado de GitHub ("Add file" → "Upload files", arrastrando el contenido de la carpeta), entren a [Streamlit Community Cloud](https://share.streamlit.io) con su cuenta de GitHub, elijan "Create app", su repositorio, la rama y `app.py`, y en "Advanced settings" la versión de Python que imprimió `exportar`. La dirección que les da es la que entregan.

## Qué hay en model/ después de exportar

| Archivo | Contenido |
|---|---|
| `pipeline.joblib` | el pipeline entrenado |
| `metadata.json` | evento, clase positiva, umbrales, decisiones, métricas, etiquetas, versiones y una ficha por columna de entrada (tipo, rango visto al entrenar, categorías y valor típico) |
| `ejemplo_entrada.csv` | tres filas inventadas con valores típicos, para que `check.py` pruebe el modelo |
| `resultados.csv` | opcional: identificador, columnas a mostrar, `probabilidad`, `decision` y `razones` |

**Las razones** de cada predicción se calculan así: para cada variable, cuánto cambia la probabilidad si toma su valor típico (la mediana o la categoría más común del entrenamiento); en el texto, cuánto cambia al quitar cada palabra. Dicen de qué depende la predicción del modelo, no qué pasaría si el cliente cambiara esa variable.

## Problemas comunes

- **"falta exportar el modelo"**: corran `exportar` con la carpeta de la app en su ruta, o copien a la app la carpeta `model/` que generó.
- **"tienen instalado scikit-learn X y el modelo se entrenó con Y"**: corran `desplegar`, que instala las versiones de `requirements.txt`.
- **GitHub no acepta un archivo de más de 25 MB desde el navegador**: reduzcan el modelo (por ejemplo, menos árboles o menos profundidad) y vuelvan a exportar.
- **"Falta la columna …"**: el pipeline espera exactamente las columnas de `X_entrenamiento`; si construyeron variables, constrúyanlas también en la tabla de resultados.
- **La app en la nube no arranca**: revisen el registro ("Manage app") en Streamlit Community Cloud; casi siempre es una versión distinta en `requirements.txt`.
