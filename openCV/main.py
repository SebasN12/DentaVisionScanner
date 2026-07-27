from src.camera import Camera


def main():

    camera = Camera("data/test_dataset")

    frames = camera.load_frames()

    print(f"Loaded {len(frames)} frames.")

    for frame in frames:
        print(frame.filename)
        print(frame.image.shape)


if __name__ == "__main__":
    main()