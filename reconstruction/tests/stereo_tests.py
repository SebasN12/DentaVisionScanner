"""
Tests for stereo reconstruction.
"""

import cv2
import numpy as np

from src.config.paths import DATASET_PATH
from src.pipeline.camera import Camera
from src.pipeline.stereo.matcher import StereoMatcher
from src.pipeline.stereo.depth import DepthReconstructor


def test_depth_reconstruction():
    """
    Tests depth reconstruction using the disparity produced
    by StereoSGBM on the Middlebury Plastic stereo dataset.

    The Middlebury 2006 dataset provides:

    - focal length: 3740 pixels
    - baseline: 160 mm

    Therefore, the resulting depth is expressed in millimeters.
    """

    camera = Camera(DATASET_PATH)

    frames = camera.load_frames()

    if len(frames) < 4:
        raise RuntimeError(
            "The Middlebury Plastic dataset must contain "
            "disp1.png, disp5.png, view1.png and view5.png."
        )

    left_frame = frames[2]
    right_frame = frames[3]

    print(f"Left image:  {left_frame.filename}")
    print(f"Right image: {right_frame.filename}")

    # ------------------------------------------------------------------
    # Compute disparity.
    # ------------------------------------------------------------------

    matcher = StereoMatcher(
        min_disparity=0,
        num_disparities=128,
        block_size=7,
    )

    disparity = matcher.compute(
        left_frame.image,
        right_frame.image,
    )

    valid_disparity = (
        np.isfinite(disparity)
        & (disparity > 0)
    )

    if not np.any(valid_disparity):
        raise RuntimeError(
            "StereoSGBM produced no valid disparity values."
        )

    # ------------------------------------------------------------------
    # Compute depth.
    # ------------------------------------------------------------------

    depth_reconstructor = DepthReconstructor(
        focal_length=3740.0,
        baseline=160.0,
    )

    depth = depth_reconstructor.compute(
        disparity
    )

    valid_depth = (
        np.isfinite(depth)
        & (depth > 0)
    )

    if not np.any(valid_depth):
        raise RuntimeError(
            "Depth reconstruction produced no valid "
            "depth values."
        )

    # ------------------------------------------------------------------
    # Print disparity statistics.
    # ------------------------------------------------------------------

    valid_disparities = disparity[valid_disparity]

    print()
    print("=" * 90)
    print("STEREO DISPARITY")
    print("=" * 90)

    print(
        f"  Shape:               "
        f"{disparity.shape}"
    )

    print(
        f"  Valid pixels:        "
        f"{len(valid_disparities)}"
    )

    print(
        f"  Minimum:             "
        f"{valid_disparities.min():.2f} px"
    )

    print(
        f"  Maximum:             "
        f"{valid_disparities.max():.2f} px"
    )

    print(
        f"  Mean:                "
        f"{valid_disparities.mean():.2f} px"
    )

    print(
        f"  Median:              "
        f"{np.median(valid_disparities):.2f} px"
    )

    # ------------------------------------------------------------------
    # Print depth statistics.
    # ------------------------------------------------------------------

    valid_depth_values = depth[valid_depth]

    print()
    print("=" * 90)
    print("DEPTH")
    print("=" * 90)

    print(
        f"  Shape:               "
        f"{depth.shape}"
    )

    print(
        f"  Valid pixels:        "
        f"{len(valid_depth_values)}"
    )

    print(
        f"  Minimum:             "
        f"{valid_depth_values.min():.2f} mm"
    )

    print(
        f"  Maximum:             "
        f"{valid_depth_values.max():.2f} mm"
    )

    print(
        f"  Mean:                "
        f"{valid_depth_values.mean():.2f} mm"
    )

    print(
        f"  Median:              "
        f"{np.median(valid_depth_values):.2f} mm"
    )

    # ------------------------------------------------------------------
    # Show representative depth values.
    # ------------------------------------------------------------------

    print()
    print("=" * 90)
    print("DISPARITY -> DEPTH")
    print("=" * 90)

    for disparity_value in (
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
    ):
        depth_value = (
            3740.0
            * 160.0
            / disparity_value
        )

        print(
            f"  Disparity {disparity_value:5.1f} px"
            f" -> "
            f"Depth {depth_value:8.2f} mm"
        )

    # ------------------------------------------------------------------
    # Visualize disparity and depth.
    # ------------------------------------------------------------------

    # --------------------------------------------------------------
    # Disparity visualization.
    # --------------------------------------------------------------

    disparity_display = np.zeros_like(
        disparity,
        dtype=np.float32,
    )

    disparity_display[valid_disparity] = (
        disparity[valid_disparity]
    )

    disparity_normalized = cv2.normalize(
        disparity_display,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    disparity_visualization = (
        disparity_normalized.astype(np.uint8)
    )

    # --------------------------------------------------------------
    # Depth visualization.
    #
    # Very small disparities produce extremely large depths.
    # These values are valid mathematically, but make the
    # visualization difficult to interpret.
    #
    # This threshold is ONLY for visualization.
    # --------------------------------------------------------------

    visualization_disparity_mask = (
        valid_disparity
        & (disparity >= 10.0)
    )

    visualization_depth = depth[
        visualization_disparity_mask
    ]

    if len(visualization_depth) == 0:
        raise RuntimeError(
            "No valid depth values available for visualization."
        )

    depth_display = np.zeros_like(
        depth,
        dtype=np.float32,
    )

    depth_display[visualization_disparity_mask] = (
        depth[visualization_disparity_mask]
    )

    depth_min = visualization_depth.min()
    depth_max = visualization_depth.max()

    depth_visualization = np.zeros_like(
        depth,
        dtype=np.uint8,
    )

    if depth_max > depth_min:

        depth_visualization[
            visualization_disparity_mask
        ] = (
            (
                (
                    depth[
                        visualization_disparity_mask
                    ]
                    - depth_min
                )
                / (depth_max - depth_min)
            )
            * 255.0
        ).astype(np.uint8)

    # --------------------------------------------------------------
    # Print visualization range.
    # --------------------------------------------------------------

    print()
    print("=" * 90)
    print("DEPTH VISUALIZATION")
    print("=" * 90)

    print(
        f"  Minimum disparity:  "
        f"{disparity[visualization_disparity_mask].min():.2f} px"
    )

    print(
        f"  Maximum disparity:  "
        f"{disparity[visualization_disparity_mask].max():.2f} px"
    )

    print(
        f"  Minimum depth:      "
        f"{depth_min:.2f} mm"
    )

    print(
        f"  Maximum depth:      "
        f"{depth_max:.2f} mm"
    )

    print(
        f"  Valid pixels:       "
        f"{len(visualization_depth)}"
    )

    # --------------------------------------------------------------
    # Show visualizations.
    # --------------------------------------------------------------

    cv2.imshow(
        "Disparity",
        disparity_visualization,
    )

    cv2.imshow(
        "Depth",
        depth_visualization,
    )

    print()
    print("Press any key to close the visualizations.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_stereo_sgbm():
    """
    Tests StereoSGBM on the Middlebury Plastic stereo dataset.

    The dataset provides view1/view5 and the corresponding
    ground-truth disparity map.
    """

    camera = Camera(DATASET_PATH)

    frames = camera.load_frames()

    if len(frames) < 4:
        raise RuntimeError(
            "The Middlebury Plastic dataset must contain "
            "disp1.png, disp5.png, view1.png and view5.png."
        )

    left_frame = frames[2]
    right_frame = frames[3]

    print(f"Left image:  {left_frame.filename}")
    print(f"Right image: {right_frame.filename}")

    # ------------------------------------------------------------------
    # Load ground truth.
    # ------------------------------------------------------------------

    disparity_path = DATASET_PATH / "disp1.png"

    ground_truth_raw = cv2.imread(
        str(disparity_path),
        cv2.IMREAD_UNCHANGED,
    )

    if ground_truth_raw is None:
        raise RuntimeError(
            f"Unable to load ground-truth disparity: "
            f"{disparity_path}"
        )

    # Middlebury third-size disparity maps use a scale factor of 3.
    ground_truth = (
        ground_truth_raw.astype(np.float32) / 3.0
    )

    valid_ground_truth = ground_truth_raw > 0

    # ------------------------------------------------------------------
    # Test different block sizes.
    # ------------------------------------------------------------------

    for block_size in (3, 5, 7):

        _evaluate_stereo(
            left_image=left_frame.image,
            right_image=right_frame.image,
            ground_truth=ground_truth,
            valid_ground_truth=valid_ground_truth,
            block_size=block_size,
        )
        

# Helpers

def _print_worst_regions(
    error_map: np.ndarray,
    valid_mask: np.ndarray,
    rows: int = 10,
    cols: int = 12,
    max_regions: int = 10,
):
    """
    Prints the regions with the highest percentage of pixels
    having an absolute disparity error above 5 pixels.
    """

    height, width = error_map.shape

    row_edges = np.linspace(
        0,
        height,
        rows + 1,
        dtype=int,
    )

    col_edges = np.linspace(
        0,
        width,
        cols + 1,
        dtype=int,
    )

    regions = []

    for row in range(rows):
        for col in range(cols):

            y0 = row_edges[row]
            y1 = row_edges[row + 1]

            x0 = col_edges[col]
            x1 = col_edges[col + 1]

            region_error = error_map[y0:y1, x0:x1]
            region_valid = valid_mask[y0:y1, x0:x1]

            valid_errors = region_error[region_valid]

            if len(valid_errors) == 0:
                continue

            regions.append(
                {
                    "row": row,
                    "col": col,
                    "x0": x0,
                    "x1": x1,
                    "y0": y0,
                    "y1": y1,
                    "valid": len(valid_errors),
                    "bad_percentage": (
                        np.mean(valid_errors > 5.0) * 100.0
                    ),
                    "mae": np.mean(valid_errors),
                    "median": np.median(valid_errors),
                }
            )

    regions.sort(
        key=lambda region: region["bad_percentage"],
        reverse=True,
    )

    print()
    print("=" * 90)
    print("WORST ERROR REGIONS")
    print("=" * 90)

    print()
    print(
        "Rank | Region | Coordinates | "
        "Bad >5px | MAE | Median | Valid"
    )

    print("-" * 90)

    for rank, region in enumerate(
        regions[:max_regions],
        start=1,
    ):
        print(
            f"{rank:4d} | "
            f"R{region['row']} C{region['col']} | "
            f"x={region['x0']}:{region['x1']}, "
            f"y={region['y0']}:{region['y1']} | "
            f"{region['bad_percentage']:7.2f}% | "
            f"{region['mae']:5.2f} | "
            f"{region['median']:6.2f} | "
            f"{region['valid']}"
        )


def _evaluate_stereo(
    left_image: np.ndarray,
    right_image: np.ndarray,
    ground_truth: np.ndarray,
    valid_ground_truth: np.ndarray,
    block_size: int,
):
    """
    Runs StereoSGBM and evaluates the resulting disparity
    against the Middlebury ground truth.
    """

    matcher = StereoMatcher(
        min_disparity=0,
        num_disparities=128,
        block_size=block_size,
    )

    disparity = matcher.compute(
        left_image,
        right_image,
    )

    valid_disparity = disparity > 0

    valid_comparison = (
        valid_ground_truth
        & valid_disparity
    )

    predicted = disparity[valid_comparison]
    expected = ground_truth[valid_comparison]

    if len(predicted) == 0:
        raise RuntimeError(
            "There are no pixels where both StereoSGBM "
            "and the ground truth provide valid disparities."
        )

    absolute_error = np.abs(
        predicted - expected
    )

    signed_error = (
        predicted - expected
    )

    error_map = np.zeros_like(
        disparity,
        dtype=np.float32,
    )

    error_map[valid_comparison] = absolute_error

    print()
    print("=" * 90)
    print(f"BLOCK SIZE: {block_size}")
    print("=" * 90)

    print()
    print("StereoSGBM:")
    print(f"  Disparity shape:    {disparity.shape}")
    print(
        f"  Minimum disparity:  "
        f"{predicted.min():.2f}"
    )
    print(
        f"  Maximum disparity:  "
        f"{predicted.max():.2f}"
    )
    print(
        f"  Mean disparity:     "
        f"{predicted.mean():.2f}"
    )
    print(
        f"  Median disparity:   "
        f"{np.median(predicted):.2f}"
    )

    print()
    print("Ground truth:")
    print(
        f"  Minimum disparity:  "
        f"{expected.min():.2f}"
    )
    print(
        f"  Maximum disparity:  "
        f"{expected.max():.2f}"
    )
    print(
        f"  Mean disparity:     "
        f"{expected.mean():.2f}"
    )
    print(
        f"  Median disparity:   "
        f"{np.median(expected):.2f}"
    )

    print()
    print("Pixel-wise comparison:")
    print(
        f"  Valid comparison pixels: "
        f"{len(predicted)}"
    )
    print(
        f"  Mean absolute error:      "
        f"{absolute_error.mean():.2f} px"
    )
    print(
        f"  Median absolute error:    "
        f"{np.median(absolute_error):.2f} px"
    )
    print(
        f"  Maximum absolute error:   "
        f"{absolute_error.max():.2f} px"
    )
    print(
        f"  Mean signed error:        "
        f"{signed_error.mean():.2f} px"
    )
    print(
        f"  Median signed error:      "
        f"{np.median(signed_error):.2f} px"
    )

    print(
        f"  Absolute error > 1 px:    "
        f"{np.mean(absolute_error > 1.0) * 100:.2f}%"
    )
    print(
        f"  Absolute error > 2 px:    "
        f"{np.mean(absolute_error > 2.0) * 100:.2f}%"
    )
    print(
        f"  Absolute error > 5 px:    "
        f"{np.mean(absolute_error > 5.0) * 100:.2f}%"
    )

    _print_worst_regions(
        error_map=error_map,
        valid_mask=valid_comparison,
    )