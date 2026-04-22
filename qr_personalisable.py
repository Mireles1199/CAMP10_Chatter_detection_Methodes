#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image, ImageColor, ImageDraw
import qrcode
from qrcode.constants import ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q, ERROR_CORRECT_H
import os


# =========================================================
# CONFIGURACIÓN
# Cambia solo estos valores
# =========================================================

CONFIG = {
    # URL o texto del QR
    "data": "https://mireles1199.github.io/CAMP10_Chatter_detection_Methodes/",

    # Nombre del archivo de salida
    "output": "maxent_qr.png",

    # Ruta del logo central (PNG recomendado). Usa None si no quieres logo.
    "logo_path": None,

    # Tamaño general
    "box_size": 18,   # tamaño de cada cuadrito
    "border": 6,      # borde blanco exterior en módulos

    # Corrección de errores: L, M, Q, H
    # H es recomendable si usarás logo
    "error_correction": "Q",

    # Colores
    "background_color": "white",     # por ejemplo: "white", "#FFFFFF"
    "module_color": "#7A1F73",       # color de los módulos normales
    "eye_outer_color": "#F2A100",    # color del borde de las esquinas
    "eye_inner_color": "#7A1F73",    # color del centro de las esquinas

    # Formas: "square", "rounded", "circle"
    "module_shape": "rounded",
    "eye_shape": "rounded",

    # Qué tan grande se dibuja cada elemento dentro de su celda
    # 1.0 = ocupa toda la celda
    "module_size_ratio": 0.96,
    "eye_size_ratio": 1.00,

    # Logo central
    "logo_scale": 0.16,   # tamaño del logo respecto al ancho del QR
    "logo_padding": 20,   # margen blanco alrededor del logo

    # True = fondo transparente
    "transparent_background": False,
}


# =========================================================
# FUNCIONES INTERNAS
# =========================================================

ERROR_MAP = {
    "L": ERROR_CORRECT_L,
    "M": ERROR_CORRECT_M,
    "Q": ERROR_CORRECT_Q,
    "H": ERROR_CORRECT_H,
}

def is_in_finder_area(row, col, size):
    anchors = [
        (0, 0),
        (0, size - 7),
        (size - 7, 0),
    ]
    for r0, c0 in anchors:
        if r0 <= row < r0 + 7 and c0 <= col < c0 + 7:
            return True
    return False


def draw_finder(draw, start_row, start_col, box_size, border, outer_color, inner_color, bg_color, shape="square"):
    x0 = (start_col + border) * box_size
    y0 = (start_row + border) * box_size

    outer_size = 7 * box_size
    middle_size = 5 * box_size
    inner_size = 3 * box_size

    outer = [x0, y0, x0 + outer_size, y0 + outer_size]
    middle = [x0 + box_size, y0 + box_size, x0 + box_size + middle_size, y0 + box_size + middle_size]
    inner = [x0 + 2 * box_size, y0 + 2 * box_size, x0 + 2 * box_size + inner_size, y0 + 2 * box_size + inner_size]

    if shape == "rounded":
        r1 = max(4, int(box_size * 1.2))
        r2 = max(4, int(box_size * 1.0))
        r3 = max(4, int(box_size * 0.8))

        draw.rounded_rectangle(outer, radius=r1, fill=outer_color)
        draw.rounded_rectangle(middle, radius=r2, fill=bg_color)
        draw.rounded_rectangle(inner, radius=r3, fill=inner_color)
    else:
        draw.rectangle(outer, fill=outer_color)
        draw.rectangle(middle, fill=bg_color)
        draw.rectangle(inner, fill=inner_color)

def parse_color(value, allow_transparent=True):
    if allow_transparent and isinstance(value, str) and value.lower() == "transparent":
        return (255, 255, 255, 0)
    rgb = ImageColor.getrgb(value)
    if len(rgb) == 3:
        return (*rgb, 255)
    return rgb


def build_qr_matrix(data, error_correction):
    qr = qrcode.QRCode(
        version=None,
        error_correction=ERROR_MAP[error_correction],
        box_size=10,
        border=0,
    )
    qr.add_data(data)
    qr.make(fit=True)
    return qr.get_matrix()


def classify_finder_cell(row, col, size):
    """
    Detecta si una celda pertenece a una esquina del QR.
    Devuelve:
    - 'outer' = borde exterior
    - 'inner' = centro
    - 'none'  = zona vacía dentro del finder
    - None    = no pertenece a las esquinas
    """
    anchors = [
        (0, 0),
        (0, size - 7),
        (size - 7, 0),
    ]

    for r0, c0 in anchors:
        if r0 <= row < r0 + 7 and c0 <= col < c0 + 7:
            rr = row - r0
            cc = col - c0

            if rr in (0, 6) or cc in (0, 6):
                return "outer"

            if 2 <= rr <= 4 and 2 <= cc <= 4:
                return "inner"

            return "none"

    return None


def draw_module(draw, x, y, box_size, color, shape, size_ratio):
    inner_size = max(1, int(round(box_size * size_ratio)))
    offset = (box_size - inner_size) / 2

    x0 = x + offset
    y0 = y + offset
    x1 = x0 + inner_size
    y1 = y0 + inner_size

    if shape == "square":
        draw.rectangle([x0, y0, x1, y1], fill=color)

    elif shape == "rounded":
        radius = max(1, int(inner_size * 0.28))
        draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=color)

    elif shape == "circle":
        draw.ellipse([x0, y0, x1, y1], fill=color)

    else:
        raise ValueError(f"Forma no soportada: {shape}")


def paste_logo(base_img, logo_path, logo_scale, logo_padding, background_rgba):
    logo = Image.open(logo_path).convert("RGBA")

    qr_w, qr_h = base_img.size
    target_w = int(qr_w * logo_scale)
    scale = target_w / logo.width
    target_h = max(1, int(logo.height * scale))

    logo = logo.resize((target_w, target_h), Image.LANCZOS)

    pad = max(0, int(logo_padding))

    card = Image.new(
        "RGBA",
        (logo.width + pad * 2, logo.height + pad * 2),
        (255, 255, 255, 0)
    )

    card_draw = ImageDraw.Draw(card)
    radius = max(8, min(card.size) // 8)
    card_draw.rounded_rectangle(
        [0, 0, card.size[0] - 1, card.size[1] - 1],
        radius=radius,
        fill=background_rgba
    )

    lx = (card.width - logo.width) // 2
    ly = (card.height - logo.height) // 2
    card.alpha_composite(logo, (lx, ly))

    x = (qr_w - card.width) // 2
    y = (qr_h - card.height) // 2
    base_img.alpha_composite(card, (x, y))


def generar_qr(config):
    data = config["data"].strip()
    if not data:
        raise ValueError("La URL o el texto del QR no puede estar vacío.")

    error_correction = config["error_correction"].upper()
    if error_correction not in ERROR_MAP:
        raise ValueError("error_correction debe ser L, M, Q o H.")

    module_shape = config["module_shape"]
    eye_shape = config["eye_shape"]

    if module_shape not in {"square", "rounded", "circle"}:
        raise ValueError("module_shape debe ser: square, rounded o circle")

    if eye_shape not in {"square", "rounded", "circle"}:
        raise ValueError("eye_shape debe ser: square, rounded o circle")

    matrix = build_qr_matrix(data, error_correction)
    qr_modules = len(matrix)

    box_size = int(config["box_size"])
    border = int(config["border"])

    total_modules = qr_modules + (border * 2)
    img_size = total_modules * box_size

    bg_rgba = parse_color(
        "transparent" if config["transparent_background"] else config["background_color"]
    )

    module_rgba = parse_color(config["module_color"])
    eye_outer_rgba = parse_color(config["eye_outer_color"])
    eye_inner_rgba = parse_color(config["eye_inner_color"])

    img = Image.new("RGBA", (img_size, img_size), bg_rgba)
    draw = ImageDraw.Draw(img)

    for row in range(qr_modules):
        for col in range(qr_modules):
            if not matrix[row][col]:
                continue

            # Los ojos se dibujan aparte, como bloques sólidos
            if is_in_finder_area(row, col, qr_modules):
                continue

            x = (col + border) * box_size
            y = (row + border) * box_size

            draw_module(
                draw,
                x,
                y,
                box_size,
                module_rgba,
                module_shape,
                float(config["module_size_ratio"])
            )

    # Dibujar los 3 ojos como formas continuas
    draw_finder(draw, 0, 0, box_size, border, eye_outer_rgba, eye_inner_rgba, bg_rgba, config["eye_shape"])
    draw_finder(draw, 0, qr_modules - 7, box_size, border, eye_outer_rgba, eye_inner_rgba, bg_rgba, config["eye_shape"])
    draw_finder(draw, qr_modules - 7, 0, box_size, border, eye_outer_rgba, eye_inner_rgba, bg_rgba, config["eye_shape"])

    logo_path = config["logo_path"]
    if logo_path:
        if not os.path.exists(logo_path):
            raise FileNotFoundError(f"No se encontró el logo: {logo_path}")

        logo_bg = parse_color("white") if config["transparent_background"] else bg_rgba

        paste_logo(
            img,
            logo_path,
            float(config["logo_scale"]),
            int(config["logo_padding"]),
            logo_bg
        )

    output = config["output"]
    output_dir = os.path.dirname(os.path.abspath(output))
    os.makedirs(output_dir, exist_ok=True)

    img.save(output)
    return output


# =========================================================
# EJECUCIÓN
# =========================================================

if __name__ == "__main__":
    archivo = generar_qr(CONFIG)
    print(f"QR generado correctamente: {archivo}")