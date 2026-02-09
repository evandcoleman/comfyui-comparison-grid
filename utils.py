"""
Utility functions for the ComparisonGrid node.

Handles label derivation, text overlay, grid assembly, and tensor/PIL conversions.
"""

import os
import re
import logging

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("comfyui-comparison-grid")

# ---------------------------------------------------------------------------
# Label derivation
# ---------------------------------------------------------------------------

def derive_label(value, param):
    """Derive a human-readable label from an axis value.

    For lora values: extracts 'Epoch N' from the filename, or uses the filename stem.
    For other params: just str(value).
    """
    if param == "lora":
        s = str(value)
        match = re.search(r"Epoch\s+(\d+)", s, re.IGNORECASE)
        if match:
            return f"Epoch {match.group(1)}"
        # Strip extension and path for a readable fallback
        stem = os.path.splitext(os.path.basename(s))[0]
        return stem
    return str(value)

# ---------------------------------------------------------------------------
# Hex color parsing
# ---------------------------------------------------------------------------

def hex_to_rgb(hex_color):
    """Convert '#RRGGBB' to (R, G, B) tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return (255, 255, 255)
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

# ---------------------------------------------------------------------------
# Tensor <-> PIL conversions
# ---------------------------------------------------------------------------

def tensor_to_pil(tensor):
    """Convert a (1, H, W, 3) float32 tensor [0,1] to a PIL Image."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def pil_to_tensor(img):
    """Convert a PIL Image to a (1, H, W, 3) float32 tensor [0,1]."""
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)

# ---------------------------------------------------------------------------
# Font loading
# ---------------------------------------------------------------------------

def load_font(font_path, font_size):
    """Load a TrueType font, falling back to PIL default."""
    if font_path:
        for candidate in [
            font_path,
            os.path.join("/usr/share/fonts/truetype", font_path),
            os.path.join("/System/Library/Fonts", font_path),
        ]:
            if os.path.isfile(candidate):
                try:
                    return ImageFont.truetype(candidate, font_size)
                except Exception:
                    pass
    try:
        return ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        return ImageFont.load_default()

# ---------------------------------------------------------------------------
# Text overlay on a single cell image
# ---------------------------------------------------------------------------

def overlay_text(pil_img, text, font, font_color_rgb, bg_color_rgb):
    """Draw a semi-transparent label bar at the bottom of the image."""
    img = pil_img.copy().convert("RGBA")
    w, h = img.size

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    bar_h = text_h + 16
    bar_y = h - bar_h

    draw.rectangle(
        [(0, bar_y), (w, h)],
        fill=(*bg_color_rgb, 180),
    )

    text_x = (w - text_w) // 2
    text_y = bar_y + (bar_h - text_h) // 2 - bbox[1]
    draw.text((text_x, text_y), text, fill=(*font_color_rgb, 255), font=font)

    composited = Image.alpha_composite(img, overlay).convert("RGB")
    return composited

# ---------------------------------------------------------------------------
# Grid assembly with row/column headers
# ---------------------------------------------------------------------------

def assemble_grid(cell_images, row_labels, col_labels, font, font_color_rgb, bg_color_rgb):
    """Assemble cell PIL images into a labeled grid.

    Args:
        cell_images: 2D list [row][col] of PIL Images (all same size)
        row_labels: list of row label strings
        col_labels: list of col label strings
        font: PIL ImageFont
        font_color_rgb: (R, G, B) for header text
        bg_color_rgb: (R, G, B) for header background

    Returns:
        PIL Image of the assembled grid.
    """
    num_rows = len(row_labels)
    num_cols = len(col_labels)
    cell_w, cell_h = cell_images[0][0].size

    header_h = 60
    header_w = 200
    padding = 4

    grid_w = header_w + num_cols * (cell_w + padding) + padding
    grid_h = header_h + num_rows * (cell_h + padding) + padding

    grid = Image.new("RGB", (grid_w, grid_h), bg_color_rgb)
    draw = ImageDraw.Draw(grid)

    for col_idx, label in enumerate(col_labels):
        x = header_w + padding + col_idx * (cell_w + padding)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = x + (cell_w - tw) // 2
        ty = (header_h - th) // 2 - bbox[1]
        draw.text((tx, ty), label, fill=font_color_rgb, font=font)

    for row_idx, label in enumerate(row_labels):
        y = header_h + padding + row_idx * (cell_h + padding)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = (header_w - tw) // 2
        ty = y + (cell_h - th) // 2 - bbox[1]
        draw.text((tx, ty), label, fill=font_color_rgb, font=font)

    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            x = header_w + padding + col_idx * (cell_w + padding)
            y = header_h + padding + row_idx * (cell_h + padding)
            grid.paste(cell_images[row_idx][col_idx], (x, y))

    return grid
