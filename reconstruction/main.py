import tests.main_tests as tests
from src.visualization.visualizer import Visualizer
from pathlib import Path
from src.config.paths import OUTPUT_PATH

def main():

    # Change this depending on what you want to test

    # Pipeline A: pairwise reconstruction with OpenCV

    # tests.load_frames()

    # tests.test_camera()

    # tests.test_features()

    # tests.test_matching()

    # tests.test_pose()

    # tests.test_triangulation()

    # tests.test_pairwise_reconstruction()

    tests.test_pairwise_reconstruction_sequence()

    # tests.test_pairwise_bundle_adjustment()


    # Pipeline B: OpenMVG reconstruction

    # visualizer = Visualizer()

    # visualizer.show_ply(Path("C:\dev\openMVG_test\output\sfm\colored_cloud.ply"))

    # visualizer.show_ply(OUTPUT_PATH / "reconstruction" / "pair_015.ply")

    # tests.test_openmvg_sparse_pipeline()

    # tests.test_openmvs_prepare_dense()

    # tests.test_openmvs_dense_pipeline()



if __name__ == "__main__":
    main()