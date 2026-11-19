"""ZONA FIJA: acompaña el despliegue de la app paso a paso. Detecta si están en Google Colab o en su computador
y sigue el camino que corresponde. No modifiquen este archivo: check.py verifica que siga igual.

En su computador (terminal, desde la carpeta de la app):
    python desplegar.py

En Google Colab (con la carpeta de la app en su Drive):
    import sys
    sys.path.append("/content/drive/MyDrive/.../plantilla_app")
    from desplegar import desplegar
    desplegar("/content/drive/MyDrive/.../plantilla_app")

Qué hace: 1) instala las librerías de requirements.txt, 2) corre check.py, 3) abre la app para probarla,
4) deja los archivos listos para GitHub y 5) explica cómo publicarla en Streamlit Community Cloud."""
import importlib.util
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import zipfile
from pathlib import Path

AQUI = Path(__file__).resolve().parent
PUERTO = 8501
LIMITE_WEB_MB = 25          # GitHub no acepta archivos más grandes subidos desde el navegador
NO_SUBIR = {"datos", "__pycache__", ".git", ".venv", "venv", ".ipynb_checkpoints", "dist"}


def entorno():
    """'colab', 'jupyter' (cuaderno en su computador) o 'terminal'."""
    try:
        hay_colab = importlib.util.find_spec("google.colab") is not None
    except (ImportError, ValueError):
        hay_colab = "google.colab" in sys.modules
    if hay_colab or "COLAB_RELEASE_TAG" in os.environ:
        return "colab"
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return "jupyter"
    except ImportError:
        pass
    return "terminal"


def titulo(texto):
    print("\n" + "=" * 70 + f"\n{texto}\n" + "=" * 70)


def instalar(carpeta):
    titulo("1. Instalar las librerías de requirements.txt")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(carpeta / "requirements.txt")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stderr[-1500:])
        raise SystemExit("No se pudieron instalar las librerías. Revisen las versiones de requirements.txt.")
    print("Listo.")


def revisar(carpeta):
    titulo("2. Revisar la app con check.py")
    r = subprocess.run([sys.executable, "check.py"], cwd=carpeta, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    print(r.stdout)
    if r.returncode != 0:
        print("Corrijan lo que dice [FALLA] y vuelvan a correr desplegar.")
        return False
    return True


def _puerto_abierto(puerto):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", puerto)) == 0


def _puerto_libre():
    for puerto in range(PUERTO, PUERTO + 20):
        if not _puerto_abierto(puerto):
            return puerto
    raise SystemExit("No encontré un puerto libre para abrir la app. Cierren otras apps de Streamlit y reintenten.")


def probar(carpeta, modo):
    """Arranca la app en segundo plano y la abre. Devuelve el proceso de Streamlit."""
    titulo("3. Probar la app")
    puerto = _puerto_libre()
    comando = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(puerto),
               "--server.headless", "true", "--browser.gatherUsageStats", "false"]
    if modo == "colab":
        comando += ["--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]
    registro = open(carpeta.parent / "registro_streamlit.txt", "w", encoding="utf-8")
    proceso = subprocess.Popen(comando, cwd=carpeta, stdout=registro, stderr=subprocess.STDOUT)
    for _ in range(90):
        if _puerto_abierto(puerto) or proceso.poll() is not None:
            break
        time.sleep(1)
    if not _puerto_abierto(puerto):
        proceso.kill()
        raise SystemExit(f"La app no arrancó. Lean {registro.name} para ver el error.")
    if modo == "colab":
        from google.colab import output
        print("Abran el enlace que aparece abajo para probar la app (funciona mientras esta sesión de Colab esté activa).")
        output.serve_kernel_port_as_window(puerto, path="/")
    else:
        url = f"http://localhost:{puerto}"
        print(f"La app está corriendo en {url}  (si no se abrió sola, copien esa dirección en el navegador).")
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass
    return proceso


def _archivos_para_subir(carpeta):
    archivos = []
    for raiz, dirs, nombres in os.walk(carpeta):
        dirs[:] = [d for d in dirs if d not in NO_SUBIR]
        for n in nombres:
            if n.endswith((".pyc", ".zip")):
                continue
            archivos.append(Path(raiz) / n)
    return archivos


def preparar_github(carpeta, modo):
    titulo("4. Archivos para GitHub")
    archivos = _archivos_para_subir(carpeta)
    grandes = [f for f in archivos if f.stat().st_size > LIMITE_WEB_MB * 1e6]
    for f in grandes:
        print(f"OJO: {f.relative_to(carpeta)} pesa {f.stat().st_size / 1e6:.0f} MB y GitHub no lo acepta desde el "
              f"navegador (máximo {LIMITE_WEB_MB} MB). Reduzcan el modelo (por ejemplo, menos árboles) y vuelvan a exportar.")
    if (carpeta / ".git").exists() and shutil.which("git") and modo != "colab":
        return _subir_con_git(carpeta)
    destino = carpeta.parent / "para_subir_a_github.zip"
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
        for f in archivos:
            z.write(f, f.relative_to(carpeta))
    print(f"Quedó {destino.name} con {len(archivos)} archivos (sin datos del curso).")
    if modo == "colab":
        try:
            from google.colab import files
            files.download(str(destino))
        except Exception:
            print(f"Descárguenlo desde la carpeta {destino.parent}.")
    print("""
Cómo subirlos a GitHub desde el navegador:
  a. Descompriman el .zip en su computador.
  b. En github.com abran su repositorio privado (o créenlo en https://github.com/new).
  c. Elijan "Add file" → "Upload files" y arrastren TODO el contenido de la carpeta descomprimida
     (las carpetas también), no la carpeta misma.
  d. Abajo, escriban un mensaje corto y presionen "Commit changes".
  Si ya habían subido una versión, suban los archivos otra vez: GitHub reemplaza los que cambiaron.""")
    return None


def _subir_con_git(carpeta):
    r = subprocess.run(["git", "status", "--short"], cwd=carpeta, capture_output=True, text=True, errors="replace")
    if not r.stdout.strip():
        print("La carpeta es un repositorio de git y no hay cambios por subir.")
        return None
    print("La carpeta es un repositorio de git. Cambios por subir:\n" + r.stdout)
    respuesta = input("¿Los subo a GitHub ahora (git add, commit y push)? [s/n] ").strip().lower()
    if respuesta != "s":
        print("No subí nada.")
        return None
    for paso in (["git", "add", "-A"], ["git", "commit", "-m", "Actualiza la app"], ["git", "push"]):
        r = subprocess.run(paso, cwd=carpeta, capture_output=True, text=True, errors="replace")
        if r.returncode != 0:
            print(r.stdout + r.stderr)
            print("Falló " + " ".join(paso) + ". Si es un problema de permisos, suban los archivos desde el navegador.")
            return None
    print("Subido.")
    return None


def instrucciones_streamlit(carpeta):
    titulo("5. Publicar en Streamlit Community Cloud")
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    try:
        import json
        meta = json.loads((carpeta / "model" / "metadata.json").read_text(encoding="utf-8"))
        version = meta.get("python", version)
    except Exception:
        pass
    print(f"""  a. Entren a https://share.streamlit.io con su cuenta de GitHub.
     La primera vez, autoricen el acceso a sus repositorios privados.
  b. Elijan "Create app" y la opción de desplegar desde un repositorio de GitHub
     (la app queda pública; el repositorio sigue privado).
  c. Repository: su repositorio. Branch: main. Main file path: app.py
  d. En "Advanced settings" elijan Python {version} (la misma versión con que exportaron el modelo).
  e. Presionen "Deploy" y esperen unos minutos. La dirección que les da es la que entregan.
  Si la app no arranca, abran "Manage app" (abajo a la derecha) y lean el final del registro:
  casi siempre es una versión de requirements.txt distinta a la que imprimió exportar.""")


def desplegar(carpeta=None, instalar_librerias=True):
    """Corre los cinco pasos del despliegue. carpeta: la carpeta de la app (por defecto, la de este archivo)."""
    try:
        sys.stdout.reconfigure(errors="replace")      # consolas de Windows que no muestran algunos símbolos
    except Exception:
        pass
    modo = entorno()
    carpeta = Path(carpeta).resolve() if carpeta else AQUI
    print(f"Entorno detectado: {'Google Colab' if modo == 'colab' else 'su computador'} "
          f"(Python {platform.python_version()}). Carpeta de la app: {carpeta}")
    if modo == "colab" and str(carpeta).startswith("/content/drive") and not Path("/content/drive/MyDrive").exists():
        from google.colab import drive
        drive.mount("/content/drive")
    if not (carpeta / "app.py").exists():
        raise SystemExit("En esa carpeta no está app.py. Indiquen la carpeta de la plantilla.")
    if instalar_librerias:
        instalar(carpeta)
    if not revisar(carpeta):
        return
    proceso = probar(carpeta, modo)
    preparar_github(carpeta, modo)
    instrucciones_streamlit(carpeta)
    if modo == "terminal" and proceso is not None:
        print("\nLa app sigue abierta para que la prueben. Cuando terminen, presionen Ctrl+C aquí.")
        try:
            proceso.wait()
        except KeyboardInterrupt:
            proceso.terminate()
            print("App cerrada.")
    return proceso


if __name__ == "__main__":
    desplegar(sys.argv[1] if len(sys.argv) > 1 else None)
