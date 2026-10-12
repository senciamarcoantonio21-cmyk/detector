"""Detector de objetos en imágenes y en vivo con Faster R-CNN."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any


def cargar_modelo() -> tuple[Any, list[str], Any, Any]:
    """Carga Faster R-CNN preentrenado con COCO."""
    try:
        import torch
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_V2_Weights,
            fasterrcnn_resnet50_fpn_v2,
        )
    except ModuleNotFoundError as error:
        if error.name not in {"torch", "torchvision"}:
            raise
        raise SystemExit(
            "Faltan dependencias. Instálalas con: "
            "python -m pip install -r requirements.txt"
        ) from error

    pesos = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    modelo = fasterrcnn_resnet50_fpn_v2(weights=pesos)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = modelo.to(dispositivo).eval()
    return modelo, pesos.meta["categories"], torch, dispositivo


def detectar(
    imagen_rgb: Any,
    modelo: Any,
    categorias: list[str],
    torch: Any,
    dispositivo: Any,
    confianza: float,
    cantidad: int,
) -> list[tuple[int, int, int, int, str, float]]:
    """Detecta objetos en una imagen RGB y devuelve cajas, nombres y confianza."""
    tensor = (
        torch.from_numpy(imagen_rgb)
        .permute(2, 0, 1)
        .float()
        .div(255)
        .to(dispositivo)
    )
    with torch.inference_mode():
        prediccion = modelo([tensor])[0]

    resultados = []
    for indice in range(len(prediccion["scores"])):
        puntuacion = prediccion["scores"][indice].item()
        if puntuacion < confianza:
            continue
        x1, y1, x2, y2 = (
            round(coordenada)
            for coordenada in prediccion["boxes"][indice].tolist()
        )
        etiqueta = categorias[prediccion["labels"][indice].item()]
        resultados.append((x1, y1, x2, y2, etiqueta, puntuacion))
        if len(resultados) >= cantidad:
            break
    return resultados


def dibujar_resultados(
    fotograma: Any,
    resultados: list[tuple[int, int, int, int, str, float]],
    cv2: Any,
) -> None:
    if not resultados:
        cv2.putText(
            fotograma,
            "No se detectan objetos conocidos",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 200, 255),
            2,
        )
        return

    for x1, y1, x2, y2, etiqueta, puntuacion in resultados:
        texto = f"{etiqueta}: {puntuacion:.0%}"
        cv2.rectangle(fotograma, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            fotograma,
            texto,
            (max(5, x1), max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 220, 0),
            2,
        )


def abrir_opencv() -> Any:
    try:
        import cv2
    except ModuleNotFoundError as error:
        if error.name != "cv2":
            raise
        raise SystemExit(
            "Falta OpenCV. Instálalo con: python -m pip install -r requirements.txt"
        ) from error
    return cv2


def usar_camara(
    numero: int,
    modelo: Any,
    categorias: list[str],
    torch: Any,
    dispositivo: Any,
    confianza: float,
    cantidad: int,
) -> None:
    cv2 = abrir_opencv()
    camara = cv2.VideoCapture(numero)
    if not camara.isOpened():
        camara.release()
        raise SystemExit(
            f"No se pudo abrir la cámara {numero}. Comprueba los permisos "
            "de cámara de Windows o prueba con --camara 1."
        )

    camara.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    titulo_ventana = "Detector de objetos - Q para salir"
    cv2.namedWindow(titulo_ventana, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(titulo_ventana, 640, 480)

    resultados: list[tuple[int, int, int, int, str, float]] = []
    proxima_deteccion = 0.0
    print(
        "Cámara activa en vista previa tipo selfie. "
        "Muestra un objeto y pulsa Q en el video para salir."
    )

    try:
        while True:
            leido, fotograma = camara.read()
            if not leido:
                raise SystemExit("No se pudo leer un fotograma de la cámara.")

            fotograma = cv2.flip(fotograma, 1)
            ahora = time.monotonic()
            if ahora >= proxima_deteccion:
                imagen_rgb = cv2.cvtColor(fotograma, cv2.COLOR_BGR2RGB)
                resultados = detectar(
                    imagen_rgb,
                    modelo,
                    categorias,
                    torch,
                    dispositivo,
                    confianza,
                    cantidad,
                )
                proxima_deteccion = ahora + 0.2

            dibujar_resultados(fotograma, resultados, cv2)
            cv2.imshow(titulo_ventana, fotograma)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camara.release()
        cv2.destroyAllWindows()


def detectar_archivo(
    ruta: Path,
    modelo: Any,
    categorias: list[str],
    torch: Any,
    dispositivo: Any,
    confianza: float,
    cantidad: int,
) -> None:
    cv2 = abrir_opencv()
    fotograma = cv2.imread(str(ruta))
    if fotograma is None:
        raise SystemExit(f"No se pudo abrir la imagen: {ruta}")

    imagen_rgb = cv2.cvtColor(fotograma, cv2.COLOR_BGR2RGB)
    resultados = detectar(
        imagen_rgb, modelo, categorias, torch, dispositivo, confianza, cantidad
    )
    dibujar_resultados(fotograma, resultados, cv2)
    destino = ruta.with_name(f"{ruta.stem}_detectada{ruta.suffix}")
    if not cv2.imwrite(str(destino), fotograma):
        raise SystemExit(f"No se pudo guardar la imagen anotada: {destino}")
    print(f"Imagen con detecciones guardada en: {destino}")
    if resultados:
        for _, _, _, _, etiqueta, puntuacion in resultados:
            print(f"{etiqueta}: {puntuacion:.0%}")
    else:
        print("No se detectaron objetos conocidos.")


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detecta y dibuja recuadros alrededor de objetos "
            "reconocidos con el modelo COCO."
        ),
        epilog=(
            "Sin argumentos, inicia la cámara integrada (número 0). "
            "La primera ejecución descarga el modelo."
        ),
    )
    parser.add_argument(
        "imagen",
        nargs="?",
        help="ruta opcional a una imagen; se guarda una copia con las detecciones",
    )
    parser.add_argument(
        "--camara",
        type=int,
        metavar="NUMERO",
        help="número de cámara; normalmente la cámara integrada es la 0",
    )
    parser.add_argument(
        "--confianza",
        type=float,
        default=0.5,
        metavar="0..1",
        help="confianza mínima para mostrar objetos (por defecto: 0.5)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        metavar="N",
        help="máximo de objetos mostrados por fotograma (por defecto: 10)",
    )
    return parser


def main(argumentos: list[str] | None = None) -> int:
    parser = crear_parser()
    opciones = parser.parse_args(argumentos)

    if opciones.imagen and opciones.camara is not None:
        parser.error("indica una imagen o --camara NUMERO, pero no ambos.")
    if opciones.camara is not None and opciones.camara < 0:
        parser.error("--camara debe ser cero o un número positivo.")
    if not 0 < opciones.confianza <= 1:
        parser.error("--confianza debe ser mayor que 0 y como máximo 1.")
    if not 1 <= opciones.top <= 100:
        parser.error("--top debe estar entre 1 y 100.")

    modelo, categorias, torch, dispositivo = cargar_modelo()
    cantidad_categorias = sum(
        categoria not in {"__background__", "N/A"} for categoria in categorias
    )
    print(
        f"Detector COCO | {cantidad_categorias} categorías de objetos | "
        f"Dispositivo: {dispositivo}"
    )

    if opciones.imagen:
        detectar_archivo(
            Path(opciones.imagen),
            modelo,
            categorias,
            torch,
            dispositivo,
            opciones.confianza,
            opciones.top,
        )
    else:
        usar_camara(
            opciones.camara if opciones.camara is not None else 0,
            modelo,
            categorias,
            torch,
            dispositivo,
            opciones.confianza,
            opciones.top,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
