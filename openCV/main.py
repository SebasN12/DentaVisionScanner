from src.pipeline.camera import Camera
from src.pipeline.features import FeatureDetector
from src.visualization.visualizer import Visualizer


def main():

    camera = Camera("data/test_dataset")

    detector = FeatureDetector()

    frames = camera.load_frames()

    print(f"Loaded {len(frames)} frames.\n")

    for frame in frames:

        detector.detect(frame)

        print(
            f"{frame.filename}: "
            f"{len(frame.keypoints)} keypoints"
        )

        output = Visualizer.draw_keypoints(
            frame,
            "output/features",
        )

        print(f"Saved: {output}")


if __name__ == "__main__":
    main()