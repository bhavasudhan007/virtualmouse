import math
from typing import Tuple, Iterable, Optional

# ...new file...
def _unpack_points(a, b=None) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if b is None:
        # a is expected to be an iterable of two points
        p1, p2 = a[0], a[1]
    else:
        p1, p2 = a, b
    return (float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))


def get_distance(a: Iterable[Tuple[float, float]], b: Optional[Tuple[float, float]] = None, scale: Optional[Tuple[int, int]] = None) -> float:
    """
    Euclidean distance between two 2D points.
    - a can be (p1, p2) or p1 and b provided.
    - If `scale` is provided as (width, height) and input points are normalized (0..1),
      the function will convert to pixel coordinates before computing distance.
    """
    p1, p2 = _unpack_points(a, b)

    if scale is not None:
        w, h = scale
        p1 = (p1[0] * w, p1[1] * h)
        p2 = (p2[0] * w, p2[1] * h)

    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def get_angle(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    """
    Returns the angle (in degrees) at p2 formed by the points p1-p2-p3.
    Works with normalized or pixel coordinates.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # Vectors p2->p1 and p2->p3
    v1 = (x1 - x2, y1 - y2)
    v2 = (x3 - x2, y3 - y2)

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])

    if mag1 == 0 or mag2 == 0:
        return 0.0

    cos_ang = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    angle_rad = math.acos(cos_ang)
    angle_deg = math.degrees(angle_rad)
    return angle_deg
# ...end of file...
