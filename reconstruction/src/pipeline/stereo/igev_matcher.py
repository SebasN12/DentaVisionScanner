"""
IGEV stereo matching through OpenStereo.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from easydict import EasyDict

import sys

from src.config.paths import OPENSTEREO_PATH


if not OPENSTEREO_PATH.exists():
    raise FileNotFoundError(
        f"OpenStereo directory not found: {OPENSTEREO_PATH}"
    )

if str(OPENSTEREO_PATH) not in sys.path:
    sys.path.insert(0, str(OPENSTEREO_PATH))


from stereo.utils import common_utils
from stereo.modeling import build_trainer
from stereo.datasets.dataset_template import build_transform_by_cfg

class IGEVMatcher:
    """
    Stereo matcher using the IGEV model provided by OpenStereo.

    The matcher receives a rectified stereo pair and returns
    a dense disparity map at the original image resolution.
    """

    def __init__(
        self,
        config_path: Path | str,
        checkpoint_path: Path | str,
    ):
        """
        Initialize the IGEV model.

        Parameters
        ----------
        config_path:
            Path to the OpenStereo IGEV configuration file.

        checkpoint_path:
            Path to the pretrained IGEV checkpoint.
        """

        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"IGEV config not found: {self.config_path}"
            )

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"IGEV checkpoint not found: {self.checkpoint_path}"
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. IGEV requires a CUDA-capable GPU "
                "for this inference pipeline."
            )

        self.device = torch.device("cuda:0")

        # --------------------------------------------------------------
        # Load OpenStereo configuration.
        # --------------------------------------------------------------

        yaml_config = common_utils.config_loader(
            str(self.config_path)
        )

        self.cfgs = EasyDict(yaml_config)

        self.cfgs.MODEL.PRETRAINED_MODEL = (
            str(self.checkpoint_path)
        )

        # --------------------------------------------------------------
        # Build IGEV model through OpenStereo.
        # --------------------------------------------------------------

        self.args = EasyDict({
            "run_mode": "infer",
            "dist_mode": False,
        })

        logger = common_utils.create_logger(
            log_file=None,
            rank=0,
        )

        self.trainer = build_trainer(
            self.args,
            self.cfgs,
            local_rank=0,
            global_rank=0,
            logger=logger,
            tb_writer=None,
        )

        self.model = self.trainer.model
        self.model.eval()

        # --------------------------------------------------------------
        # Build the same evaluation transform used by OpenStereo.
        # --------------------------------------------------------------

        transform_config = (
            self.cfgs.DATA_CONFIG
            .DATA_TRANSFORM
            .EVALUATING
        )

        self.transform = build_transform_by_cfg(
            transform_config
        )

    @torch.no_grad()
    def compute(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
    ) -> np.ndarray:
        """
        Compute dense stereo disparity.

        Parameters
        ----------
        left_image:
            Left rectified image in BGR format.

        right_image:
            Right rectified image in BGR format.

        Returns
        -------
        np.ndarray
            Dense disparity map with the same height and width
            as the input images.
        """

        if left_image is None:
            raise ValueError(
                "left_image must not be None."
            )

        if right_image is None:
            raise ValueError(
                "right_image must not be None."
            )

        if left_image.ndim != 3:
            raise ValueError(
                "left_image must be a 3-channel image."
            )

        if right_image.ndim != 3:
            raise ValueError(
                "right_image must be a 3-channel image."
            )

        if left_image.shape != right_image.shape:
            raise ValueError(
                "Left and right images must have the same shape."
            )

        original_height, original_width = (
            left_image.shape[:2]
        )

        # OpenStereo's inference script loads images using PIL
        # and therefore expects RGB images.
        left_rgb = left_image[:, :, ::-1]
        right_rgb = right_image[:, :, ::-1]

        left_rgb = np.ascontiguousarray(left_rgb)
        right_rgb = np.ascontiguousarray(right_rgb)

        sample = {
            "left": left_rgb.astype(np.float32),
            "right": right_rgb.astype(np.float32),
        }

        # --------------------------------------------------------------
        # OpenStereo preprocessing.
        #
        # The configured transform may pad the image because IGEV
        # operates on dimensions compatible with its feature hierarchy.
        # --------------------------------------------------------------

        sample = self.transform(sample)

        # Add batch dimension.
        sample["left"] = sample["left"].unsqueeze(0)
        sample["right"] = sample["right"].unsqueeze(0)

        # Move tensors to GPU.
        for key, value in sample.items():
            if torch.is_tensor(value):
                sample[key] = value.to(self.device)

        # --------------------------------------------------------------
        # IGEV inference.
        # --------------------------------------------------------------

        with torch.amp.autocast(
            device_type="cuda",
            enabled=self.cfgs.OPTIMIZATION.AMP,
        ):
            model_pred = self.model(sample)

        disparity = (
            model_pred["disp_pred"]
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        # --------------------------------------------------------------
        # Remove OpenStereo padding.
        #
        # RightTopPad adds pixels to the top and right of the image.
        # Therefore the original image corresponds to the upper-left
        # region after removing the added top/right dimensions.
        # --------------------------------------------------------------

        output_height, output_width = disparity.shape

        pad_height = output_height - original_height
        pad_width = output_width - original_width

        if pad_height < 0 or pad_width < 0:
            raise RuntimeError(
                "IGEV output is smaller than the original image: "
                f"output={disparity.shape}, "
                f"original={(original_height, original_width)}"
            )

        disparity = disparity[
            pad_height:
            pad_height + original_height,
            0:original_width,
        ]

        return disparity.astype(
            np.float32,
            copy=False,
        )