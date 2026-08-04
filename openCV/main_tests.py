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

from src.config.camera import CAMERA_MATRIX

from src.config.paths import (
    DATASET_PATH,
    FEATURES_OUTPUT,
    MATCHES_OUTPUT,
    INLIERS_OUTPUT,
    POSES_OUTPUT_FILE,
    RECONSTRUCTION_OUTPUT_FILE
)



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

    reconstructor = Reconstructor(
        triangulator
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

    reconstructor = Reconstructor(
        triangulator
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
    # Write result
    #
    output = PointCloudWriter.write_ply(
        cloud,
        RECONSTRUCTION_OUTPUT_FILE,
    )


    print(
        f"Saved: {output}"
    )