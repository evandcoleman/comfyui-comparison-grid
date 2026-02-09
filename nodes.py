"""
ComparisonGrid node for ComfyUI.

Replaces the entire multi-node comparison grid workflow with a single node
that iterates over row/col axis values, sampling and labeling each cell internally.

LoRA files are loaded from ComfyUI's standard loras folder via folder_paths.
Use the ModelManagerLoRADownload node upstream to download LoRAs and connect
them to this node's lora_name or lora_N inputs.
"""

import logging

import torch
import numpy as np

from .utils import (
    derive_label,
    hex_to_rgb,
    load_font,
    overlay_text,
    assemble_grid,
    tensor_to_pil,
    pil_to_tensor,
)

logger = logging.getLogger("comfyui-comparison-grid")


class AnyType(str):
    """Matches any ComfyUI type for flexible inputs."""
    def __ne__(self, __value):
        return False

any_type = AnyType("*")


class FlexibleOptionalInputType(dict):
    """Dict that accepts any key, returning (any_type,) for unknowns.
    Used to accept dynamic lora_N inputs."""
    def __init__(self, base=None):
        super().__init__(base or {})
    def __contains__(self, key):
        return True
    def __getitem__(self, key):
        if key in self.keys():
            return super().__getitem__(key)
        return (any_type,)


def _parse_axis_values(raw_text, param):
    """Parse multiline text into a list of typed values.

    One value per line. Blank lines are skipped.
    Returns list of (label, value) tuples.
    """
    lines = [line.strip() for line in raw_text.strip().splitlines() if line.strip()]
    results = []
    for line in lines:
        label = derive_label(line, param)
        if param in ("strength", "cfg", "denoise"):
            value = float(line)
        elif param in ("seed", "steps"):
            value = int(line)
        elif param in ("sampler", "scheduler", "lora"):
            value = line
        else:
            value = line
        results.append((label, value))
    return results


class ComparisonGrid:
    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("cell_images", "grid_image", "num_rows", "num_cols")
    FUNCTION = "execute"
    CATEGORY = "image/comparison"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers
        import folder_paths

        axis_choices = ["lora", "strength", "seed", "steps", "cfg", "sampler", "scheduler", "denoise"]
        lora_list = ["none"] + folder_paths.get_filename_list("loras")

        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),

                "row_axis": (axis_choices,),
                "row_values": ("STRING", {"multiline": True, "default": ""}),
                "col_axis": (axis_choices,),
                "col_values": ("STRING", {"multiline": True, "default": ""}),

                "default_lora": (lora_list,),
                "default_strength": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.05}),
                "default_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "default_steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "default_cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "default_sampler": (comfy.samplers.KSampler.SAMPLERS,),
                "default_scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "default_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),

                "font_size": ("INT", {"default": 36, "min": 8, "max": 256}),
                "font_color": ("STRING", {"default": "#FFFFFF"}),
                "bg_color": ("STRING", {"default": "#000000"}),
            },
            "optional": FlexibleOptionalInputType({
                "font_path": ("STRING", {"default": ""}),
                "lora_name": (any_type,),
            }),
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def execute(
        self,
        model,
        clip,
        vae,
        positive,
        negative,
        latent_image,
        row_axis,
        row_values,
        col_axis,
        col_values,
        default_lora,
        default_strength,
        default_seed,
        default_steps,
        default_cfg,
        default_sampler,
        default_scheduler,
        default_denoise,
        font_size,
        font_color,
        bg_color,
        font_path="",
        lora_name=None,
        **kwargs,
    ):
        import comfy.sample
        import comfy.sd
        import comfy.utils
        import folder_paths

        if row_axis == col_axis:
            raise ValueError(f"Row and column axes must differ (both are '{row_axis}')")

        # Collect lora filenames from linked inputs (lora_name + lora_N kwargs)
        # These are filenames output by ModelManagerLoRADownload nodes.
        # When lora is an axis, these replace the multiline text values.
        linked_loras = []
        if lora_name is not None:
            linked_loras.append(str(lora_name))
        for key in sorted(kwargs.keys()):
            if key.startswith("lora_") and kwargs[key] is not None:
                linked_loras.append(str(kwargs[key]))

        # When lora is an axis and we have linked lora inputs, use those as axis values
        if linked_loras:
            if row_axis == "lora":
                row_items = [(derive_label(v, "lora"), v) for v in linked_loras]
            else:
                row_items = _parse_axis_values(row_values, row_axis)

            if col_axis == "lora":
                col_items = [(derive_label(v, "lora"), v) for v in linked_loras]
            else:
                col_items = _parse_axis_values(col_values, col_axis)
        else:
            row_items = _parse_axis_values(row_values, row_axis)
            col_items = _parse_axis_values(col_values, col_axis)

        if not row_items:
            raise ValueError("No row values provided")
        if not col_items:
            raise ValueError("No col values provided")

        num_rows = len(row_items)
        num_cols = len(col_items)
        total_cells = num_rows * num_cols

        # For the non-axis default lora: use lora_name link if available, else the combo selection
        if row_axis != "lora" and col_axis != "lora":
            effective_default_lora = linked_loras[0] if linked_loras else default_lora
        else:
            effective_default_lora = default_lora
        if effective_default_lora == "none":
            effective_default_lora = ""

        defaults = {
            "lora": effective_default_lora,
            "strength": default_strength,
            "seed": default_seed,
            "steps": default_steps,
            "cfg": default_cfg,
            "sampler": default_sampler,
            "scheduler": default_scheduler,
            "denoise": default_denoise,
        }

        font_color_rgb = hex_to_rgb(font_color)
        bg_color_rgb = hex_to_rgb(bg_color)
        font = load_font(font_path, font_size)

        # Pre-load LoRA weights from ComfyUI's standard loras folder
        lora_cache = {}
        lora_names_needed = set()

        if row_axis == "lora":
            for _, v in row_items:
                lora_names_needed.add(v)
        if col_axis == "lora":
            for _, v in col_items:
                lora_names_needed.add(v)
        if row_axis != "lora" and col_axis != "lora" and effective_default_lora:
            lora_names_needed.add(effective_default_lora)

        for name in lora_names_needed:
            lora_path = folder_paths.get_full_path_or_raise("loras", name)
            lora_cache[name] = comfy.utils.load_torch_file(lora_path, safe_load=True)

        # Prepare latent
        latent_samples = latent_image["samples"]
        noise_mask = latent_image.get("noise_mask")

        latent_samples = comfy.sample.fix_empty_latent_channels(model, latent_samples)

        pbar = comfy.utils.ProgressBar(total_cells)
        cell_tensors = []
        cell_pil_grid = [[None] * num_cols for _ in range(num_rows)]
        row_labels = [label for label, _ in row_items]
        col_labels = [label for label, _ in col_items]

        for row_idx, (row_label, row_value) in enumerate(row_items):
            for col_idx, (col_label, col_value) in enumerate(col_items):
                params = dict(defaults)
                params[row_axis] = row_value
                params[col_axis] = col_value

                cur_lora = params["lora"]
                strength = float(params["strength"])
                seed = int(params["seed"])
                steps = int(params["steps"])
                cfg = float(params["cfg"])
                sampler_name = str(params["sampler"])
                scheduler = str(params["scheduler"])
                denoise = float(params["denoise"])

                # Apply LoRA
                if cur_lora and cur_lora in lora_cache:
                    lora_data = lora_cache[cur_lora]
                    model_lora, clip_lora = comfy.sd.load_lora_for_models(
                        model, clip, lora_data, strength, strength
                    )
                else:
                    model_lora, clip_lora = model, clip

                # Prepare noise
                noise = comfy.sample.prepare_noise(latent_samples, seed)

                # Sample
                samples = comfy.sample.sample(
                    model_lora,
                    noise,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    positive,
                    negative,
                    latent_samples,
                    denoise=denoise,
                    disable_noise=False,
                    force_full_denoise=True,
                    noise_mask=noise_mask,
                    callback=None,
                    disable_pbar=True,
                    seed=seed,
                )

                # Decode
                image_tensor = vae.decode(samples)

                # Overlay text
                pil_img = tensor_to_pil(image_tensor)
                label_text = f"{row_label} | {col_label}"
                pil_labeled = overlay_text(pil_img, label_text, font, font_color_rgb, bg_color_rgb)

                cell_tensors.append(pil_to_tensor(pil_labeled))
                cell_pil_grid[row_idx][col_idx] = pil_labeled

                pbar.update(1)

        # Assemble outputs
        cell_batch = torch.cat(cell_tensors, dim=0)

        grid_pil = assemble_grid(
            cell_pil_grid, row_labels, col_labels, font, font_color_rgb, bg_color_rgb
        )
        grid_tensor = pil_to_tensor(grid_pil)

        return (cell_batch, grid_tensor, num_rows, num_cols)
