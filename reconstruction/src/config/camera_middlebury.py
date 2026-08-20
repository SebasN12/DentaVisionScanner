"""
Camera calibration parameters for the Middlebury 2006
stereo dataset (ThirdSize).
"""

import numpy as np


# ---------------------------------------------------------------------
# Middlebury 2006 Stereo Dataset
#
# Images:
#   ThirdSize
#   Resolution: 423 x 370
#
# The original Middlebury dataset provides:
#   Focal length: 3740 px (FullSize)
#   Baseline:     160 mm
#
# Since the images used here are ThirdSize, the focal length is
# scaled by 1/3.
#
# The images are already rectified and radial distortion has been
# removed by the dataset authors.
# ---------------------------------------------------------------------

FOCAL_LENGTH = 3740.0 / 3.0

PRINCIPAL_POINT = (
    423.0 / 2.0,
    370.0 / 2.0,
)

BASELINE_MM = 160.0


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


REPROJECTION_MATRIX = np.array(
    [
        [
            1.0,
            0.0,
            0.0,
            -PRINCIPAL_POINT[0],
        ],
        [
            0.0,
            1.0,
            0.0,
            -PRINCIPAL_POINT[1],
        ],
        [
            0.0,
            0.0,
            0.0,
            FOCAL_LENGTH,
        ],
        [
            0.0,
            0.0,
            1.0 / BASELINE_MM,
            0.0,
        ],
    ],
    dtype=np.float64,
)