"""
Camera calibration parameters.
"""

import numpy as np

# ---------------------------------------------------------------------
# Approximate intrinsic parameters for the Palm_Desert_Micro dataset.
#
# These values are only used until the real camera calibration is
# available.
# ---------------------------------------------------------------------

FOCAL_LENGTH = 2500.0

PRINCIPAL_POINT = (
    2000.0,
    1500.0,
)

CAMERA_MATRIX = np.array(
    [
        [FOCAL_LENGTH, 0.0, PRINCIPAL_POINT[0]],
        [0.0, FOCAL_LENGTH, PRINCIPAL_POINT[1]],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

DISTORTION_COEFFICIENTS = np.zeros(
    5,
    dtype=np.float64,
)