"""
Helper module for DIP Lab Task 04 - Geometric Transformations with Custom Inputs
Author : Syed Muhammad Hassan Gillani | Roll No: 2024-SE-42 | 5th Semester (SE)

Public API
----------
load_image(path)                     -> RGB uint8 array (validated)
transform(image, operation, **p)     -> transformed RGB image (single reusable entry point)
ask_parameters / run_interactive     -> prompt loops with input correction
translation_matrix / rotation_matrix -> the 2x3 affine matrices used internally

Conventions (declared once, used everywhere)
--------------------------------------------
* Colour : images are loaded with OpenCV (BGR) and converted ONCE to RGB; all
           processing and display use RGB uint8 arrays.
* Image coordinates are (x, y) = (column, row), origin at the TOP-LEFT, y grows
  downward. NumPy indexing is img[row, column] = img[y, x].
* Exposed areas created by translation / rotation are filled with black (0).
* Positive rotation angle = COUNTER-CLOCKWISE as seen on screen.
* Maximum output size: MAX_SIDE pixels per side and MAX_PIXELS pixels in total.
"""
import math
import os
import numbers

import cv2
import numpy as np

# ----------------------------------------------------------------- configuration
MAX_SIDE = 4000            # max width or height of any output image
MAX_PIXELS = 16_000_000    # max width*height of any output image
BACKGROUND = 0             # fill value for exposed areas (black)

INTERPOLATIONS = {
    "nearest":  cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic":  cv2.INTER_CUBIC,
    "lanczos":  cv2.INTER_LANCZOS4,
}
FLIP_DIRECTIONS = {"horizontal": 1, "vertical": 0, "both": -1}   # cv2.flip codes

# operation -> (required parameter names, in prompt order)
OPERATIONS = {
    "scale":     ("sx", "sy", "interpolation"),
    "crop":      ("left", "top", "right", "bottom"),
    "flip":      ("direction",),
    "translate": ("tx", "ty"),
    "rotate":    ("angle", "expand"),
}


class TransformError(ValueError):
    """Raised for any invalid image, operation name or parameter."""


# ----------------------------------------------------------------- image loading
def load_image(path):
    """Load an image file as RGB uint8. Validates existence, decoding and size."""
    if not isinstance(path, (str, os.PathLike)):
        raise TransformError("Image path must be a string or path-like object.")
    if not os.path.exists(path):
        raise TransformError(f"File not found: {path!r}")
    if not os.path.isfile(path):
        raise TransformError(f"Not a file: {path!r}")
    bgr = cv2.imread(os.fspath(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise TransformError(f"File exists but could not be decoded as an image: {path!r}")
    if bgr.ndim != 3 or bgr.shape[0] == 0 or bgr.shape[1] == 0:
        raise TransformError(f"Decoded image has zero/invalid dimensions: {bgr.shape}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)          # consistent colour handling


def _check_image(image):
    if not isinstance(image, np.ndarray):
        raise TransformError("image must be a NumPy array.")
    if image.dtype != np.uint8:
        raise TransformError(f"image must be uint8, got {image.dtype}.")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (1, 3, 4)):
        raise TransformError(f"Unsupported image shape {image.shape}.")
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise TransformError(f"Image has an empty dimension: {image.shape}.")


# ----------------------------------------------------------------- validators
def _is_number(v):
    return isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_))


def _finite(value, name):
    if not _is_number(value):
        raise TransformError(f"'{name}' must be a number, got {value!r} ({type(value).__name__}).")
    v = float(value)
    if math.isnan(v) or math.isinf(v):
        raise TransformError(f"'{name}' must be finite (not NaN/infinity), got {value!r}.")
    return v


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TransformError(f"'{name}' must be an integer, got {value!r} ({type(value).__name__}).")
    return int(value)


def _check_output_size(w, h, what):
    if w < 1 or h < 1:
        raise TransformError(f"{what} would give an empty image ({w}x{h}); both sides must be >= 1 pixel.")
    if w > MAX_SIDE or h > MAX_SIDE or w * h > MAX_PIXELS:
        raise TransformError(f"{what} would give {w}x{h} px, above the limit "
                             f"(max {MAX_SIDE} px per side, {MAX_PIXELS:,} px total).")


def _round_half_up(x):
    return int(math.floor(x + 0.5))


# ----------------------------------------------------------------- affine matrices
def translation_matrix(tx, ty):
    """2x3 affine matrix that moves content right by tx and down by ty."""
    tx, ty = _finite(tx, "tx"), _finite(ty, "ty")
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float64)


def rotation_matrix(width, height, angle, expand=False):
    """
    2x3 affine matrix rotating about the image centre ((w-1)/2, (h-1)/2).
    Positive angle = counter-clockwise on screen. Returns (matrix, out_w, out_h).
    """
    angle = _finite(angle, "angle")
    if not isinstance(expand, (bool, np.bool_)):
        raise TransformError(f"'expand' must be True or False, got {expand!r}.")
    a = angle % 360.0
    rad = math.radians(a)
    c, s = math.cos(rad), math.sin(rad)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    # OpenCV-style CCW matrix for a y-down coordinate system
    M = np.array([[c,  s, (1 - c) * cx - s * cy],
                  [-s, c, s * cx + (1 - c) * cy]], dtype=np.float64)
    out_w, out_h = width, height
    if expand:
        out_w = int(math.ceil(width * abs(c) + height * abs(s) - 1e-9))
        out_h = int(math.ceil(width * abs(s) + height * abs(c) - 1e-9))
        _check_output_size(out_w, out_h, "Expanded rotation")
        M[0, 2] += (out_w - 1) / 2.0 - cx
        M[1, 2] += (out_h - 1) / 2.0 - cy
    return M, out_w, out_h


# ----------------------------------------------------------------- the entry point
def transform(image, operation, **params):
    """
    Single reusable entry point.

    operation / required parameters
      'scale'     sx, sy (finite > 0), interpolation in nearest|bilinear|bicubic|lanczos
      'crop'      left, top, right, bottom (ints; 0<=left<right<=W, 0<=top<bottom<=H; right/bottom exclusive)
      'flip'      direction in horizontal|vertical|both
      'translate' tx, ty (finite numbers, may be negative or zero), canvas size unchanged
      'rotate'    angle (finite degrees, + = counter-clockwise), expand (bool)

    Returns a NEW array; the input image is never modified.
    Raises TransformError for anything invalid (unknown operation, missing/extra
    parameters, bad values, empty or oversized output).
    """
    _check_image(image)
    if not isinstance(operation, str):
        raise TransformError(f"Operation must be a string, got {type(operation).__name__}.")
    op = operation.strip().lower()
    if op not in OPERATIONS:
        raise TransformError(f"Unknown operation {operation!r}. Supported: {', '.join(OPERATIONS)}.")
    required = OPERATIONS[op]
    missing = [p for p in required if p not in params]
    if missing:
        raise TransformError(f"Operation '{op}' is missing required parameter(s): {', '.join(missing)}.")
    extra = [p for p in params if p not in required]
    if extra:
        raise TransformError(f"Operation '{op}' got unexpected parameter(s): {', '.join(extra)}.")

    h, w = image.shape[:2]

    if op == "scale":
        sx, sy = _finite(params["sx"], "sx"), _finite(params["sy"], "sy")
        if sx <= 0 or sy <= 0:
            raise TransformError(f"Scale factors must be greater than zero, got sx={sx}, sy={sy}.")
        interp = params["interpolation"]
        if not isinstance(interp, str) or interp.strip().lower() not in INTERPOLATIONS:
            raise TransformError(f"Unsupported interpolation {interp!r}. "
                                 f"Choose from: {', '.join(INTERPOLATIONS)}.")
        if w * sx > 1e9 or h * sy > 1e9:      # avoid giant ints before the size check
            raise TransformError(f"Scale factors sx={sx}, sy={sy} are far too large "
                                 f"(limit {MAX_SIDE} px per side).")
        out_w, out_h = _round_half_up(w * sx), _round_half_up(h * sy)
        _check_output_size(out_w, out_h, f"Scaling by sx={sx}, sy={sy}")
        return cv2.resize(image, (out_w, out_h), interpolation=INTERPOLATIONS[interp.strip().lower()])

    if op == "crop":
        l, t = _integer(params["left"], "left"), _integer(params["top"], "top")
        r, b = _integer(params["right"], "right"), _integer(params["bottom"], "bottom")
        if not (0 <= l < r <= w):
            raise TransformError(f"Invalid horizontal range: need 0 <= left < right <= width ({w}); "
                                 f"got left={l}, right={r}.")
        if not (0 <= t < b <= h):
            raise TransformError(f"Invalid vertical range: need 0 <= top < bottom <= height ({h}); "
                                 f"got top={t}, bottom={b}.")
        return image[t:b, l:r].copy()                       # right/bottom exclusive

    if op == "flip":
        d = params["direction"]
        if not isinstance(d, str) or d.strip().lower() not in FLIP_DIRECTIONS:
            raise TransformError(f"Unsupported flip direction {d!r}. "
                                 f"Choose from: {', '.join(FLIP_DIRECTIONS)}.")
        return cv2.flip(image, FLIP_DIRECTIONS[d.strip().lower()])

    if op == "translate":
        M = translation_matrix(params["tx"], params["ty"])    # validates tx, ty
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=BACKGROUND)

    # rotate
    M, out_w, out_h = rotation_matrix(w, h, params["angle"], params["expand"])
    return cv2.warpAffine(image, M, (out_w, out_h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=BACKGROUND)


# ----------------------------------------------------------------- text -> value parsers
def _clean(text, name):
    if not isinstance(text, str):
        raise TransformError(f"'{name}': expected text input.")
    t = text.strip()
    if t == "":
        raise TransformError(f"'{name}': input is empty - a value is required (no default is assumed).")
    return t


def parse_float(text, name):
    t = _clean(text, name)
    if "_" in t:
        raise TransformError(f"'{name}': '{t}' is not a valid number.")
    try:
        v = float(t)
    except ValueError:
        raise TransformError(f"'{name}': '{t}' is not a number.") from None
    if math.isnan(v) or math.isinf(v):
        raise TransformError(f"'{name}': NaN/infinity is not allowed (got '{t}').")
    return v


def parse_int(text, name):
    t = _clean(text, name)
    try:
        return int(t)
    except ValueError:
        raise TransformError(f"'{name}': '{t}' is not a whole number (integer required).") from None


def parse_choice(text, name, choices):
    t = _clean(text, name).lower()
    if t not in choices:
        raise TransformError(f"'{name}': '{t}' is not supported. Choose from: {', '.join(choices)}.")
    return t


def parse_bool(text, name):
    t = _clean(text, name).lower()
    if t in ("y", "yes", "true", "1"):
        return True
    if t in ("n", "no", "false", "0"):
        return False
    raise TransformError(f"'{name}': '{t}' is not valid. Answer yes or no.")


_PARAM_PARSERS = {
    "sx": lambda t: parse_float(t, "sx"),
    "sy": lambda t: parse_float(t, "sy"),
    "interpolation": lambda t: parse_choice(t, "interpolation", list(INTERPOLATIONS)),
    "left": lambda t: parse_int(t, "left"),
    "top": lambda t: parse_int(t, "top"),
    "right": lambda t: parse_int(t, "right"),
    "bottom": lambda t: parse_int(t, "bottom"),
    "direction": lambda t: parse_choice(t, "direction", list(FLIP_DIRECTIONS)),
    "tx": lambda t: parse_float(t, "tx"),
    "ty": lambda t: parse_float(t, "ty"),
    "angle": lambda t: parse_float(t, "angle"),
    "expand": lambda t: parse_bool(t, "expand"),
}
_PROMPTS = {
    "sx": "Horizontal scale factor sx (> 0)", "sy": "Vertical scale factor sy (> 0)",
    "interpolation": "Interpolation (nearest/bilinear/bicubic/lanczos)",
    "left": "Left x (int)", "top": "Top y (int)", "right": "Right x, exclusive (int)",
    "bottom": "Bottom y, exclusive (int)",
    "direction": "Flip direction (horizontal/vertical/both)",
    "tx": "Horizontal offset tx in px (+ = right)", "ty": "Vertical offset ty in px (+ = down)",
    "angle": "Angle in degrees (+ = counter-clockwise)", "expand": "Expand canvas to fit? (yes/no)",
}


# ----------------------------------------------------------------- interactive loops
class ScriptedInput:
    """Feeds pre-recorded answers to the prompt loop (used so the notebook runs unattended).
    Echoes each answer like a terminal would. Raises EOFError when answers run out."""
    def __init__(self, answers):
        self.answers = list(answers)

    def __call__(self, prompt=""):
        if not self.answers:
            raise EOFError("scripted input exhausted")
        a = self.answers.pop(0)
        print(f"{prompt}{a}")
        return a


def ask_parameters(operation, input_fn=input, max_tries=5):
    """Prompt for every parameter of `operation`; re-ask a parameter until valid."""
    params = {}
    for name in OPERATIONS[operation]:
        for attempt in range(max_tries):
            raw = input_fn(f"  {_PROMPTS[name]}: ")
            try:
                params[name] = _PARAM_PARSERS[name](raw)
                break
            except TransformError as e:
                print(f"    [!] {e}  (try again)")
        else:
            raise TransformError(f"Too many invalid attempts for '{name}'.")
    return params


def run_interactive(image, input_fn=input, max_tries=5):
    """
    Ask for an operation and its parameters, call transform(), and return
    (operation, params, result). Invalid entries produce a message and a re-prompt,
    never an unhandled crash. Returns None if the user quits (or input ends).
    """
    for _ in range(max_tries):
        try:
            op = input_fn(f"Operation ({'/'.join(OPERATIONS)}, or q to quit): ").strip().lower()
        except EOFError:
            print("[input ended]")
            return None
        if op in ("q", "quit"):
            return None
        if op not in OPERATIONS:
            print(f"[!] Unknown operation '{op}'. Supported: {', '.join(OPERATIONS)}.")
            continue
        try:
            params = ask_parameters(op, input_fn, max_tries)
            result = transform(image, op, **params)
            return op, params, result
        except TransformError as e:              # cross-field problems (e.g. right <= left, too big)
            print(f"[!] {e}\n    Please re-enter the parameters.")
        except EOFError:
            print("[input ended]")
            return None
    print("[!] Too many invalid attempts - stopping.")
    return None
