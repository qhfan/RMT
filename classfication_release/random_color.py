from colorsys import hsv_to_rgb
from math import exp, sqrt
import random


def get_random_colors(n, h_range=(0.0, 1.0), s_range=(0.2, 1.0), v_range=(0.6, 1.0)):
    """
    get random, visually distinct colors, for use in for example qualitative choropleth map.
    Colors are generated in the HSV space and converted to RGB values.
    Parameters:
    n: Number of colors to generate 
    h_range: Tuple of min and max value of the Hue parameter in the range 0.0-1.0
    s_range: Tuple of min and max value of the Saturation parameter in the range 0.0-1.0. 
             The min value should be a least 0.2 for good results
    v_range: Tuple of min and max value of the Value parameter in the range 0.0-1.0
             The min value should be a least 0.4 for good results
        
    Returns:
    rbg_int: a list of integer value RGB tuples
    """

    assert (
        h_range[0] >= 0 and h_range[1] <= 1
    ), "h_range min and max must be between 0-1"
    assert (
        s_range[0] >= 0 and s_range[1] <= 1
    ), "s_range min and max must be between 0-1"
    assert (
        v_range[0] >= 0 and v_range[1] <= 1
    ), "v_range min and max must be between 0-1"

    assert h_range[0] < h_range[1], "h_range min must be less than h_range max"
    assert s_range[0] < s_range[1], "s_range min must be less than s_range max"
    assert v_range[0] < v_range[1], "v_range min must be less than v_range max"

    sample_space = [
        (random.uniform(*h_range), random.uniform(*s_range), random.uniform(*v_range))
        for r in range(10 * n)
    ]
    selected_colors = []
    selected_colors.append(sample_space.pop())
    # find the color in our sample space that maximizes the minimum 
    # distance too all the already select colors.
    for i in range(n - 1):
        min_dist = [_min_dist_to_selected(selected_colors, c) for c in sample_space]
        next_color_idx = min_dist.index(max(min_dist))
        selected_colors.append(sample_space.pop(next_color_idx))
    rgb_float = (hsv_to_rgb(*s) for s in selected_colors)
    rgb_int = [
        (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))
        for r, g, b in rgb_float
    ]
    return rgb_int


def _min_dist_to_selected(selected, c):
    return min([Dvr(c, s) for s in selected])


def _dh(h1, h2):
    return min(abs(h1 - h2), 1 - abs(h1 - h2)) * 2


def _Dc(c1, c2):
    h1, s1, v1 = c1
    h2, s2, v2 = c2
    return sqrt(_dh(h1, h2) ** 2 + (s1 - s2) ** 2 + (v1 - v2) ** 2)


def Dvr(c1, c2):
    h1, s1, v1 = c1
    h2, s2, v2 = c2
    color_diff = max(_dh(h1, h2), abs(s1 - s2)) ** 2 + _Dc(c1, c2) ** 2
    return min(sqrt(color_diff / 2), 1)