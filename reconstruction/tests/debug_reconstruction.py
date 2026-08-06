"""
Debug utilities for SfM reconstruction.
"""

import numpy as np
import cv2

def debug_camera_poses(
    reconstruction,
):

    print()
    print("==========================")
    print("CAMERA POSES")
    print("==========================")


    for name, pose in reconstruction.camera_poses.items():

        print(name)

        print(
            "translation:",
            pose.translation.reshape(3)
        )

        det = np.linalg.det(
            pose.rotation
        )

        if abs(det - 1.0) > 1e-3:
            print(
                "WARNING rotation determinant:",
                det
            )

        print()


def debug_landmarks(
    reconstruction,
):

    print()
    print("==========================")
    print("LANDMARKS")
    print("==========================")


    observation_counts = []


    for landmark in reconstruction.landmarks.values():

        count = len(
            landmark.observations
        )

        observation_counts.append(
            count
        )


    observation_counts = np.asarray(
        observation_counts
    )


    print(
        "Total landmarks:",
        len(observation_counts)
    )


    print(
        "Mean observations:",
        np.mean(observation_counts)
    )


    print(
        "Median observations:",
        np.median(observation_counts)
    )


    print(
        "Min observations:",
        np.min(observation_counts)
    )


    print(
        "Max observations:",
        np.max(observation_counts)
    )


def debug_single_landmark(
    reconstruction,
    landmark_id,
):


    landmark = reconstruction.landmarks.get(
        landmark_id
    )


    if landmark is None:

        print(
            "Landmark not found"
        )

        return


    print()
    print("==========================")
    print("LANDMARK DEBUG")
    print("==========================")


    print(
        "ID:",
        landmark.id
    )


    print(
        "Track:",
        landmark.track_id
    )


    print(
        "Position:",
        landmark.position
    )


    print(
        "Observations:",
        len(landmark.observations)
    )


    for obs in landmark.observations:

        print()

        print(
            "Frame:",
            obs.frame_name
        )

        print(
            "Camera:",
            obs.camera_id
        )

        print(
            "Keypoint:",
            obs.keypoint_index
        )

        print(
            "Image point:",
            obs.image_point
        )



def find_bad_landmarks(
    reconstruction,
    errors,
    threshold=100,
):

    print()
    print("==========================")
    print("BAD LANDMARKS")
    print("==========================")


    bad_indices = np.where(
        errors > threshold
    )[0]


    print(
        "Bad observations:",
        len(bad_indices)
    )


    return bad_indices

def debug_duplicate_tracks(reconstruction):

    print()
    print("==========================")
    print("DUPLICATE TRACK CHECK")
    print("==========================")


    signatures = {}

    duplicates = 0


    for landmark in reconstruction.landmarks.values():

        obs = tuple(
            sorted(
                (
                    o.frame_name,
                    o.keypoint_index,
                )
                for o in landmark.observations
            )
        )


        if obs in signatures:

            print(
                "Duplicate landmarks:",
                signatures[obs],
                landmark.id
            )

            print(
                "Observations:",
                obs
            )

            duplicates += 1


            if duplicates >= 10:
                break

        else:

            signatures[obs] = landmark.id


    print(
        "Duplicates found:",
        duplicates
    )

def debug_bad_observations(
    reconstruction,
    camera_matrix,
    threshold=10,
    max_print=5,
):
    """
    Prints observations with large reprojection error.
    """

    camera_lookup = {
        camera_id: frame_name
        for frame_name, camera_id
        in reconstruction.camera_ids.items()
    }


    printed = 0
    bad_count = 0


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


            rvec, _ = cv2.Rodrigues(
                pose.rotation
            )


            projected, _ = cv2.projectPoints(
                point,
                rvec,
                pose.translation,
                camera_matrix,
                None,
            )


            projected = projected.reshape(2)


            error = np.linalg.norm(
                projected
                -
                observation.image_point
            )


            if error > threshold:

                bad_count += 1


                if printed < max_print:

                    print()
                    print(
                        "BAD OBSERVATION"
                    )

                    print(
                        "Landmark:",
                        landmark.id
                    )

                    print(
                        "Track:",
                        landmark.track_id
                    )

                    print(
                        "Frame:",
                        frame_name
                    )

                    print(
                        "Error:",
                        error
                    )

                    print(
                        "Observed:",
                        observation.image_point
                    )

                    print(
                        "Projected:",
                        projected
                    )

                    print(
                        "Position:",
                        landmark.position
                    )

                    printed += 1


    print()
    print(
        "Bad observations:",
        bad_count
    )

def remove_bad_observations(
    reconstruction,
    camera_matrix,
    threshold=10.0,
):
    """
    Removes observations with high reprojection error.
    """

    camera_lookup = {
        camera_id: frame_name
        for frame_name, camera_id
        in reconstruction.camera_ids.items()
    }


    removed = 0


    for landmark in reconstruction.landmarks.values():

        valid = []


        point = landmark.position.reshape(
            1,
            3
        )


        for observation in landmark.observations:


            frame_name = camera_lookup[
                observation.camera_id
            ]


            pose = reconstruction.camera_poses[
                frame_name
            ]


            rvec, _ = cv2.Rodrigues(
                pose.rotation
            )


            projected, _ = cv2.projectPoints(
                point,
                rvec,
                pose.translation,
                camera_matrix,
                None
            )


            projected = projected.reshape(2)


            error = np.linalg.norm(
                projected -
                observation.image_point
            )


            if error <= threshold:

                valid.append(
                    observation
                )

            else:

                removed += 1


        landmark.observations = valid


    print(
        "Removed observations:",
        removed
    )

def remove_empty_landmarks(
    reconstruction,
):

    removed = 0


    ids_to_remove = []


    for landmark_id, landmark in reconstruction.landmarks.items():

        if len(landmark.observations) < 2:

            ids_to_remove.append(
                landmark_id
            )


    for landmark_id in ids_to_remove:

        del reconstruction.landmarks[
            landmark_id
        ]

        removed += 1


    print(
        "Removed landmarks:",
        removed
    )

def filter_short_tracks(
    reconstruction,
    minimum_observations=3,
):

    removed = 0


    ids = []


    for landmark_id, landmark in reconstruction.landmarks.items():

        if len(landmark.observations) < minimum_observations:

            ids.append(
                landmark_id
            )


    for landmark_id in ids:

        del reconstruction.landmarks[
            landmark_id
        ]

        removed += 1


    print(
        "Removed short tracks:",
        removed
    )