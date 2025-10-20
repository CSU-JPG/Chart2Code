import numpy as np
from matplotlib import colors as mcolors
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from typing import Any, Optional, List, Dict

# Monkey patch for np.asscalar, which was removed in NumPy 1.16.
# The `colormath` library may rely on this deprecated function.
def patch_asscalar(a):
    return a.item()
setattr(np, "asscalar", patch_asscalar)


def convert_color_to_hex(color: Any) -> Optional[str]:
    """
    Safely converts various color formats into a standardized hex string.
    This function uses Matplotlib's robust converter, which can handle formats
    like color names (e.g., 'red'), RGBA tuples/lists, and NumPy arrays.

    """
    try:
        if color is None:
            return None
        # Let Matplotlib's robust converter handle the conversion.
        return mcolors.to_hex(color).upper()
    except (ValueError, TypeError):
        # If Matplotlib cannot convert it, the format is unsupported.
        return None


def convert_hex_to_lab(hex_color: str) -> LabColor:
    """
    Converts a hex color string to the CIE Lab color space.
    """
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    rgb = sRGBColor(r, g, b, is_upscaled=True)
    return convert_color(rgb, LabColor)


def calculate_color_similarity(color1: str, color2: str) -> float:
    """
    Calculates the perceptual similarity between two hex colors.

    Similarity is based on the CIEDE2000 color difference formula, which models
    human color perception. The result is normalized to a 0.0-1.0 scale, where
    1.0 means the colors are identical.

    """
    if color1.startswith('#') and color2.startswith('#'):

        lab1 = convert_hex_to_lab(color1)
        lab2 = convert_hex_to_lab(color2)
        delta = delta_e_cie2000(lab1, lab2)
        
        return max(0.0, 1.0 - delta / 100.0)

    elif not color1.startswith('#') and not color2.startswith('#'):
        return 1.0 if color1.lower() == color2.lower() else 0.0
    
    # If formats are mixed (e.g., one hex, one name), they are not comparable.
    else:
        return 0.0


def color_grouping(color_list: List[str]) -> Dict[str, List[str]]:
    """
    Groups colors by their associated chart type from a formatted list.

    This function expects each string in the input list to be formatted as
    "chart_type--#HEXCOLOR".
    """
    color_dict = {}
    for item in color_list:
        try:
            chart_type, color = item.split("--", 1)
            if chart_type not in color_dict:
                color_dict[chart_type] = []
            color_dict[chart_type].append(color)
        except ValueError:
            continue
    return color_dict

