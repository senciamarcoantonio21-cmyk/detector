# Reconocimiento de objetos con inteligencia artificial

Este programa usa **Faster R-CNN**, preentrenado con COCO, para detectar y
dibujar recuadros alrededor de varios objetos reconocidos en la cámara o en una
imagen. Muestra el nombre y la confianza de cada detección.

> El modelo reconoce las categorías incluidas en COCO, no cualquier objeto
> imaginable. Puede no reconocer objetos fuera de sus categorías o muy pequeños.

## Requisitos e instalación

Se recomienda Python 3.10 a 3.13. Desde la carpeta del proyecto, ejecuta:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

La instalación de PyTorch puede depender de si tu equipo tiene una GPU NVIDIA.
Si tienes problemas con la instalación, sigue las instrucciones oficiales de
PyTorch para seleccionar la versión compatible con tu sistema.

## Usar la cámara web

Pulsa **Ejecutar** en VS Code o ejecuta el programa sin argumentos: se abrirá
una ventana pequeña tipo selfie con la imagen espejada y los recuadros de los
objetos detectados. La cámara integrada normalmente es la número 0.

```powershell
python proyecto.py
```

También puedes indicar la cámara explícitamente:

```powershell
python proyecto.py --camara 0
```

Pulsa **Q** dentro de la ventana de video para cerrar. Si tienes más de una
cámara, prueba con `--camara 1`.

## Usar con una imagen

Guarda una copia con los recuadros junto al archivo original:

```powershell
python proyecto.py "C:\ruta\a\mi_foto.jpg"
```

Puedes ajustar el umbral de confianza (por defecto, 0.5) y el máximo de objetos:

```powershell
python proyecto.py --confianza 0.3 --top 10
```

La primera ejecución descarga el modelo y necesita conexión a internet. Las
siguientes ejecuciones usan la copia local.
