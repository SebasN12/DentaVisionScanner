from src.visualization.visualizer import Visualizer

from src_v2.reconstruction.openmvg import OpenMVG
from src_v2.reconstruction.openmvs import OpenMVS
from src_v2.reconstruction.pipeline import ReconstructionPipeline

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