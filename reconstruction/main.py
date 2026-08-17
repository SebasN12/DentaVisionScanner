import tests.modules_tests as module
import tests.pair_tests as pair
import tests.sfm_tests as sfm
from src.visualization.visualizer import Visualizer
from pathlib import Path
from src.config.paths import OUTPUT_PATH

def main():

    # Change this depending on what you want to test

    # Pipeline A: pairwise reconstruction with OpenCV

    # module.test_load_single_frame()

    # module.load_frames()

    # module.test_camera()

    # module.test_features()

    # module.test_matching()

    # module.test_pose()

    # module.test_triangulation()

    pair.test_pairwise_reconstruction()

    # pair.test_pairwise_reconstruction_sequence()

    # module.test_pairwise_bundle_adjustment()

    # pair.test_dense_reconstruction()


    # Pipeline B: OpenMVG reconstruction

    # visualizer = Visualizer()

    # visualizer.show_ply(Path("C:\dev\openMVG_test\output\sfm\colored_cloud.ply"))

    # visualizer.show_ply(OUTPUT_PATH / "reconstruction" / "pair_015.ply")

    # sfm.test_openmvg_sparse_pipeline()

    # sfm.test_openmvs_prepare_dense()

    # sfm.test_openmvs_dense_pipeline()



if __name__ == "__main__":
    main()