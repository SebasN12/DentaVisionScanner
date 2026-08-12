from xml.parsers.expat import errors

import cv2
import numpy as np

from src.pipeline.reconstructor import Reconstructor
from src.pipeline.camera import Camera
from src.pipeline.features import FeatureDetector
from src.pipeline.matching import FeatureMatcher
from src.pipeline.pose import PoseEstimator
from src.pipeline.triangulation import Triangulator
from src.visualization.visualizer import Visualizer
from src.io.point_cloud_writer import PointCloudWriter
from src.io.pose_writer import PoseWriter
from src.pipeline.track_builder import TrackBuilder
from src.pipeline.landmark_manager import LandmarkManager
from src.optimization.bundle_adjustment import BundleAdjustment
from src.core.reconstruction import Reconstruction
from src.core.landmark import Landmark
import tests.debug_reconstruction as debug


from src_v2.reconstruction.openmvg import OpenMVG
from src_v2.reconstruction.openmvs import OpenMVS
from src_v2.reconstruction.pipeline import ReconstructionPipeline

from src.config.camera import CAMERA_MATRIX

from src.config.paths import (
    DATASET_PATH,
    FEATURES_OUTPUT,
    MATCHES_OUTPUT,
    INLIERS_OUTPUT,
    POSES_OUTPUT_FILE,
    RECONSTRUCTION_OUTPUT_FILE
)


# Pipeline A: pairwise reconstruction with OpenCV

def load_frames():

    camera = Camera(DATASET_PATH)

    frames = camera.load_frames()

    print(
        f"Loaded {len(frames)} frames.\n"
    )

    return frames


def test_camera():

    frames = load_frames()

    for frame in frames:

        print(frame.filename)



def test_features():

    frames = load_frames()

    detector = FeatureDetector()
    detector.detect_sequence(frames)

    for frame in frames:

        print(
            f"{frame.filename}: "
            f"{len(frame.keypoints)} keypoints"
        )

        output = Visualizer.draw_keypoints(
            frame,
            FEATURES_OUTPUT,
        )

        print(f"Saved: {output}")



def test_matching():

    frames = load_frames()

    detector = FeatureDetector()
    detector.detect_sequence(frames)

    matcher = FeatureMatcher()
    results = matcher.match_sequence(frames)

    for result in results:

        print(
            f"{result.frame1.filename} "
            f"<-> "
            f"{result.frame2.filename}: "
            f"{len(result.good_matches)} matches"
        )

        output = Visualizer.draw_matches(
            result,
            MATCHES_OUTPUT,
        )

        print(f"Saved: {output}")

def test_pose():

    frames = load_frames()

    detector = FeatureDetector()
    detector.detect_sequence(frames)

    matcher = FeatureMatcher()
    results = matcher.match_sequence(frames)

    estimator = PoseEstimator()

    for result in results:

        estimator.estimate(
            result,
            CAMERA_MATRIX,
        )

        print(
            f"{result.frame1.filename}"
            f" -> "
            f"{result.frame2.filename}"
        )

        print(
            f"Good matches: "
            f"{len(result.good_matches)}"
        )

        print(
            f"RANSAC inliers: "
            f"{result.ransac_mask.sum()}"
        )

        print(
            f"Pose inliers: "
            f"{len(result.inlier_matches)}"
        )

        print()

        Visualizer.draw_matches(
            result,
            INLIERS_OUTPUT,
            use_inliers=True,
        )

def test_triangulation():

    frames = load_frames()

    detector = FeatureDetector()
    detector.detect_sequence(frames)

    matcher = FeatureMatcher()
    results = matcher.match_sequence(frames)

    estimator = PoseEstimator()

    triangulator = Triangulator()

    for result in results:

        estimator.estimate(
            result,
            CAMERA_MATRIX,
        )

        cloud = triangulator.triangulate(
            result,
            CAMERA_MATRIX,
        )

        print(
            f"{result.frame1.filename}"
            f" -> "
            f"{result.frame2.filename}"
        )

        print(
            f"3D points: "
            f"{len(cloud.points)}"
        )

        Visualizer.show_point_cloud(
            cloud,
        )

def test_reconstruction():

    frames = load_frames()

    detector = FeatureDetector()

    detector.detect_sequence(
        frames
    )

    matcher = FeatureMatcher()

    results = matcher.match_sequence(
        frames
    )

    print("\nMatching statistics:")

    for result in results:

        print(
            f"{result.frame1.filename} "
            f"<-> "
            f"{result.frame2.filename}"
        )

        print(
            f"  Good matches: "
            f"{len(result.good_matches)}"
        )

    print()

    estimator = PoseEstimator()

    for result in results:

        estimator.estimate(
            result,
            CAMERA_MATRIX,
        )

        print(
            f"{result.frame1.filename} "
            f"-> "
            f"{result.frame2.filename}"
        )

        print(
            f"  Matches: "
            f"{len(result.good_matches)}"
        )

        print(
            f"  Pose inliers: "
            f"{len(result.inlier_matches)}"
        )

    print()

    triangulator = Triangulator()

    track_builder = TrackBuilder()

    landmark_manager = LandmarkManager()

    reconstructor = Reconstructor(
        triangulator,
        track_builder,
        landmark_manager,
    )

    reconstruction = reconstructor.reconstruct(
        results,
        CAMERA_MATRIX,
    )

    print(
        "\nReconstruction finished."
    )

    print(
        "Reconstruction statistics:"
    )

    print(
        f"  Camera poses: "
        f"{len(reconstruction.camera_poses)}"
    )


    if reconstruction.point_cloud is not None:

        print(
            f"3D points: "
            f"{len(reconstruction.point_cloud.points)}"
        )

        Visualizer.show_point_cloud(
            reconstruction.point_cloud
        )

    else:

        print(
            "No point cloud generated."
        )


def test_sequential_reconstruction():

    frames = load_frames()


    detector = FeatureDetector()

    matcher = FeatureMatcher()

    estimator = PoseEstimator()

    triangulator = Triangulator()

    track_builder = TrackBuilder()

    landmark_manager = LandmarkManager()

    reconstructor = Reconstructor(
        triangulator,
        track_builder,
        landmark_manager,
    )


    previous_frame = None


    for current_frame in frames:


        #
        # New frame arrives
        #
        detector.detect(
            current_frame
        )


        print(
            f"Processing: "
            f"{current_frame.filename}"
        )


        #
        # First frame only initializes
        #
        if previous_frame is None:

            previous_frame = current_frame

            continue



        #
        # Match previous frame with current frame
        #
        result = matcher.match(
            previous_frame,
            current_frame,
        )


        print(
            f"Matches: "
            f"{len(result.good_matches)}"
        )



        #
        # Estimate camera movement
        #
        estimator.estimate(
            result,
            CAMERA_MATRIX,
        )


        print(
            f"Pose estimated"
        )



        #
        # Add pair to global reconstruction
        #
        reconstructor.update_pair(
            result,
            CAMERA_MATRIX,
        )


        print(
            "Added to reconstruction"
        )


        print()


        #
        # Current frame becomes previous frame
        #
        previous_frame = current_frame



    #
    # Final reconstruction
    #
    reconstruction = (
        reconstructor.get_reconstruction()
    )


    cloud = reconstruction.point_cloud


    print(
        f"Total 3D points: "
        f"{len(cloud.points)}"
    )



    #
    # Export point cloud
    #
    output = PointCloudWriter.write_ply(
        cloud,
        RECONSTRUCTION_OUTPUT_FILE,
    )


    print(
        f"Saved point cloud: {output}"
    )



    #
    # Export camera poses
    #
    pose_output = PoseWriter.write_json(
        reconstruction.camera_poses,
        POSES_OUTPUT_FILE,
    )


    print(
        f"Saved poses: {pose_output}"
    )



    #
    # Visualization
    #
    Visualizer.show_point_cloud(
        cloud
    )

def test_sequential_reconstruction_with_tracks():

    frames = load_frames()


    detector = FeatureDetector()

    matcher = FeatureMatcher()

    estimator = PoseEstimator()

    triangulator = Triangulator()

    track_builder = TrackBuilder()

    landmark_manager = LandmarkManager()


    reconstructor = Reconstructor(
        triangulator,
        track_builder,
        landmark_manager,
    )



    previous_frame = None



    for current_frame in frames:


        detector.detect(
            current_frame
        )


        print(
            f"\nProcessing: {current_frame.filename}"
        )


        #
        # First frame only initializes sequence
        #
        if previous_frame is None:

            previous_frame = current_frame

            continue



        #
        # Match consecutive frames
        #
        result = matcher.match(
            previous_frame,
            current_frame,
        )


        #
        # Estimate relative camera pose
        #
        estimator.estimate(
            result,
            CAMERA_MATRIX,
        )


        #
        # Integrate pair into reconstruction
        #
        reconstructor.update_pair(
            result,
            CAMERA_MATRIX,
        )



        #
        # Debug tracks
        #
        tracks = (
            track_builder.tracks.values()
        )


        print(
            f"Tracks: {len(track_builder.tracks)}"
        )


        if len(track_builder.tracks) > 0:

            lengths = [
                track.length
                for track in tracks
            ]

            print(
                f"Track observations - "
                f"min: {min(lengths)}, "
                f"max: {max(lengths)}, "
                f"avg: {sum(lengths)/len(lengths):.2f}"
            )



        #
        # Debug landmarks
        #
        print(
            f"Landmarks: "
            f"{len(reconstructor.reconstruction.landmarks)}"
        )


        print(
            "Added pair: "
            f"{previous_frame.filename}"
            f" -> "
            f"{current_frame.filename}"
        )


        previous_frame = current_frame



    #
    # Final reconstruction
    #
    reconstruction = (
        reconstructor.get_reconstruction()
    )


    cloud = reconstruction.point_cloud


    if cloud is None:

        raise RuntimeError(
            "No point cloud was generated."
        )



    print(
        "\nFinal reconstruction"
    )


    print(
        f"Final points: {len(cloud.points)}"
    )


    print(
        f"Final landmarks: "
        f"{len(reconstruction.landmarks)}"
    )


    print(
        f"Final cameras: "
        f"{len(reconstruction.camera_poses)}"
    )



    #
    # Check landmark observations
    #
    landmarks = (
        reconstruction.landmarks.values()
    )


    observation_lengths = [
        len(landmark.observations)
        for landmark in landmarks
    ]


    if observation_lengths:

        print(
            f"Landmark observations - "
            f"min: {min(observation_lengths)}, "
            f"max: {max(observation_lengths)}, "
            f"avg: "
            f"{sum(observation_lengths)/len(observation_lengths):.2f}"
        )



    #
    # Check tracks without landmarks
    #
    tracks_without_landmark = [
        track.id
        for track in track_builder.tracks.values()
        if track.landmark_id is None
    ]


    print(
        f"Tracks without landmark: "
        f"{len(tracks_without_landmark)}"
    )


    if len(tracks_without_landmark) > 0:

        print(
            "Example missing track ids:",
            tracks_without_landmark[:10]
        )



    #
    # Consistency check
    #
    print(
        f"Track/Landmark consistency: "
        f"{len(track_builder.tracks)} tracks "
        f"vs "
        f"{len(reconstruction.landmarks)} landmarks"
    )

    #
    # Verify camera IDs
    #
    print(
        f"Camera IDs: "
        f"{len(reconstruction.camera_ids)}"
    )

    assert len(
        reconstruction.camera_ids
    ) == len(
        reconstruction.camera_poses
    )

    #
    # Verify observations contain camera IDs
    #
    for landmark in reconstruction.landmarks.values():

        for observation in landmark.observations:

            assert observation.camera_id in reconstruction.camera_ids.values()

    print(
        "All observations have valid camera IDs."
    )
    

    #
    # Write result
    #
    output = PointCloudWriter.write_ply(
        cloud,
        RECONSTRUCTION_OUTPUT_FILE,
    )


    print(
        f"Saved: {output}"
    )


def test_bundle_adjustment():

    reconstruction = build_reconstruction()


    print_reconstruction_summary(
        reconstruction
    )


    debug.debug_duplicate_tracks(
        reconstruction
    )


    debug.debug_camera_poses(
        reconstruction
    )


    debug.debug_landmarks(
        reconstruction
    )

    debug.remove_bad_observations(
        reconstruction,
        CAMERA_MATRIX,
        threshold=10,
    )

    errors_clean = compute_reprojection_errors(
        reconstruction,
        CAMERA_MATRIX,
    )

    print_error_statistics(
        errors_clean,
        "After filtering"
    )


    debug.debug_bad_observations(
        reconstruction,
        CAMERA_MATRIX,
        threshold=10,
    )

    debug.remove_empty_landmarks(
        reconstruction)
    
    debug.filter_short_tracks(reconstruction, minimum_observations=3)


    ba = BundleAdjustment(
        CAMERA_MATRIX
    )


    print()
    print(
        "Running Bundle Adjustment..."
    )


    ba.optimize(
        reconstruction
    )


    errors_after = compute_reprojection_errors(
        reconstruction,
        CAMERA_MATRIX,
    )


    print_error_statistics(
        errors_after,
        "After BA"
    )


    _update_point_cloud_from_landmarks(
        reconstruction
    )


    Visualizer.show_point_cloud(
        reconstruction.point_cloud
    )


def build_reconstruction():

    frames = load_frames()


    detector = FeatureDetector()
    matcher = FeatureMatcher()
    estimator = PoseEstimator()
    triangulator = Triangulator()
    track_builder = TrackBuilder()
    landmark_manager = LandmarkManager()


    reconstructor = Reconstructor(
        triangulator,
        track_builder,
        landmark_manager,
    )


    previous_frame = None


    for frame in frames:

        detector.detect(frame)


        if previous_frame is None:

            previous_frame = frame
            continue


        result = matcher.match(
            previous_frame,
            frame,
        )


        estimator.estimate(
            result,
            CAMERA_MATRIX,
        )


        reconstructor.update_pair(
            result,
            CAMERA_MATRIX,
        )


        previous_frame = frame


    return reconstructor.get_reconstruction()

def print_reconstruction_summary(
    reconstruction,
):

    print()
    print("==========================")
    print("RECONSTRUCTION SUMMARY")
    print("==========================")


    print(
        "Cameras:",
        len(reconstruction.camera_poses)
    )


    print(
        "Landmarks:",
        len(reconstruction.landmarks)
    )

    observations = [
        len(lm.observations)
        for lm in reconstruction.landmarks.values()
    ]


    print(
        "Average observations per landmark:",
        np.mean(observations)
    )

    print(
        "Min observations:",
        np.min(observations)
    )

    print(
        "Max observations:",
        np.max(observations)
    )


    print(
        "Observations:",
        sum(observations)
    )


    print("==========================")

def compute_reprojection_errors(
    reconstruction,
    camera_matrix,
):
    """
    Computes reprojection errors for all observations.

    Returns:
        np.ndarray containing one error per observation.
    """

    errors = []

    frame_errors = {}


    camera_lookup = {
        camera_id: frame_name
        for frame_name, camera_id
        in reconstruction.camera_ids.items()
    }


    for landmark in reconstruction.landmarks.values():

        point = landmark.position.reshape(
            1,
            3,
        )


        for observation in landmark.observations:


            frame_name = camera_lookup.get(
                observation.camera_id
            )


            if frame_name is None:
                continue


            pose = reconstruction.camera_poses[
                frame_name
            ]


            R = pose.rotation
            t = pose.translation


            rvec, _ = cv2.Rodrigues(
                R
            )


            projected, _ = cv2.projectPoints(
                point,
                rvec,
                t,
                camera_matrix,
                None,
            )


            projected = projected.reshape(
                2
            )


            error = np.linalg.norm(
                projected
                -
                observation.image_point
            )


            errors.append(
                error
            )


            frame_errors.setdefault(
                frame_name,
                []
            ).append(
                error
            )


    print()

    print(
        "=========================="
    )

    print(
        "ERROR PER CAMERA"
    )

    print(
        "=========================="
    )


    for frame_name in sorted(frame_errors):

        values = np.asarray(
            frame_errors[frame_name]
        )


        print(
            f"{frame_name}: "
            f"mean={values.mean():8.2f}px   "
            f"median={np.median(values):8.2f}px   "
            f"count={len(values)}"
        )


    print(
        "=========================="
    )


    return np.asarray(
        errors
    )


def print_error_statistics(
    errors,
    name,
):
    if len(errors) == 0:
        print("No reprojection errors available")
        return

    print()
    print("==========================")
    print(name)
    print("==========================")


    print(
        "Mean:",
        np.mean(errors)
    )

    print(
        "Median:",
        np.median(errors)
    )

    print(
        "95 percentile:",
        np.percentile(errors,95)
    )

    print(
        "Max:",
        np.max(errors)
    )

    print("==========================")


def _update_point_cloud_from_landmarks(
    reconstruction,
):
    """
    Updates point cloud positions after optimization.
    """

    if reconstruction.point_cloud is None:
        return


    points = []


    for landmark in reconstruction.landmarks.values():

        points.append(
            landmark.position
        )


    reconstruction.point_cloud.points = np.asarray(
        points
    )

# Pipeline B: OpenMVG reconstruction

def test_openmvg_sparse_pipeline():
    openmvg = OpenMVG()

    pipeline = ReconstructionPipeline(
        openmvg=openmvg,
        clean_output=True,
    )

    point_cloud = pipeline.run_sparse()

    print("\nSparse point cloud generated at:")
    print(point_cloud)

    visualizer = Visualizer()
    visualizer.show_ply(point_cloud)

def test_openmvs_prepare_dense():
    openmvg = OpenMVG()
    openmvs = OpenMVS()

    pipeline = ReconstructionPipeline(
        openmvg=openmvg,
        openmvs=openmvs,
        clean_output=True,
    )

    scene = pipeline.prepare_dense()

    print("\nOpenMVS scene generated at:")
    print(scene)

def test_openmvs_dense_pipeline():
    openmvs = OpenMVS()

    pipeline = ReconstructionPipeline(
        openmvs=openmvs,
        clean_output=True,
    )

    point_cloud = pipeline.run_dense()

    print("\nDense point cloud generated at:")
    print(point_cloud)

    visualizer = Visualizer()
    visualizer.show_ply(point_cloud)