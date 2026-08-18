import cv2
from tqdm import tqdm

from src.config.settings import (
    ENABLE_PROGRESS_BAR,
    LOWE_RATIO,
)
from src.core.frame import Frame
from src.core.match_result import MatchResult


class FeatureMatcher:
    """
    Matches SIFT descriptors between two frames.
    """

    def __init__(self):

        self.matcher = cv2.BFMatcher()

    def match(
        self,
        frame1: Frame,
        frame2: Frame,
    ) -> MatchResult:

        raw_matches = self.matcher.knnMatch(
            frame1.descriptors,
            frame2.descriptors,
            k=2,
        )

        good_matches = []

        for first, second in raw_matches:

            if first.distance < LOWE_RATIO * second.distance:
                good_matches.append(first)

        return MatchResult(
            frame1=frame1,
            frame2=frame2,
            good_matches=good_matches,
        )
    
    def match_sequence(
        self,
        frames: list[Frame],
    ) -> list[MatchResult]:
        """
        Matches consecutive frames in a sequence.
        """

        results = []

        iterator = (
            tqdm(
                range(len(frames) - 1),
                desc="Matching frames",
            )
            if ENABLE_PROGRESS_BAR
            else range(len(frames) - 1)
        )

        for i in iterator:

            results.append(
                self.match(
                    frames[i],
                    frames[i + 1],
                )
            )

        return results