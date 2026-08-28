import tests.modules_tests as module
import tests.pair_tests as pair
import tests.sfm_tests as sfm
import tests.stereo_tests as stereo
from src.visualization.visualizer import Visualizer
from pathlib import Path
from src.config.paths import OUTPUT_PATH

def main():

    # Change this depending on what you want to test


    # Pipeline A: pairwise reconstruction with OpenCV

    # Feature based

    # module.test_load_single_frame()

    # module.load_frames()

    # module.test_camera()

    # module.test_features()

    # module.test_matching()

    # module.test_pose()

    # module.test_triangulation()

    # pair.test_pairwise_reconstruction()

    # pair.test_pairwise_reconstruction_sequence()

    # module.test_pairwise_bundle_adjustment()

    # pair.test_dense_reconstruction()

    # Plastic tests

    # pair.test_pairwise_reconstruction("view1.png", "view5.png")

    
    
    # Stereo

    # stereo.test_stereo_sgbm()

    # stereo.test_depth_reconstruction()

    # stereo.test_stereo_validator()

    # stereo.test_stereo_reconstruction()

    stereo.test_igev_disparity()

    # stereo.test_igev_reconstruction()



    # Pipeline B: OpenMVG reconstruction

    # visualizer = Visualizer()

    # visualizer.show_ply(Path("C:\dev\openMVG_test\output\sfm\colored_cloud.ply"))

    # visualizer.show_ply(OUTPUT_PATH / "reconstruction" / "pair_015.ply")

    # sfm.test_openmvg_sparse_pipeline()

    # sfm.test_openmvs_prepare_dense()

    # sfm.test_openmvs_dense_pipeline()



if __name__ == "__main__":
    main()