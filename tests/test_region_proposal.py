"""Synthetic regression tests for classical anomaly proposals."""

from pathlib import Path
import unittest

import cv2
import numpy as np

from feature_extraction import extract_feature_maps
from region_proposal import propose_regions


class RegionProposalSyntheticTests(unittest.TestCase):
    def _run(self, image: np.ndarray, stem: str):
        return propose_regions(image, extract_feature_maps(image), stem, min_area=30, max_regions=12)

    def _assert_overlaps(self, result, target, minimum=0.12):
        def iou(a, b):
            inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            return inter/max((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter,1)
        self.assertGreaterEqual(max((iou(p.bbox,target) for p in result.proposals),default=0),minimum)

    def test_long_irregular_weld(self):
        image=np.full((300,500,3),145,np.uint8)
        points=np.array([[30,135],[100,118],[190,142],[280,120],[380,139],[470,115],[470,180],[350,166],[250,180],[150,155],[30,178]])
        cv2.fillPoly(image,[points],(82,94,108)); cv2.polylines(image,[points],True,(210,215,220),4)
        rng=np.random.default_rng(2); image=np.clip(image.astype(np.int16)+rng.normal(0,9,image.shape),0,255).astype(np.uint8)
        self._assert_overlaps(self._run(image,"synthetic_weld"),(25,110,475,185),.20)

    def test_thin_crack(self):
        image=np.full((300,500,3),165,np.uint8); cv2.line(image,(35,250),(460,45),(30,30,30),4)
        self._assert_overlaps(self._run(image,"synthetic_crack"),(30,40,465,255),.08)

    def test_pitting_spots(self):
        image=np.full((300,500,3),160,np.uint8)
        for center in [(150,130),(175,155),(205,125),(340,190)]: cv2.circle(image,center,12,(55,65,75),-1)
        result=self._run(image,"synthetic_pits"); self.assertGreaterEqual(len(result.proposals),1)

    def test_large_colour_change(self):
        image=np.full((300,500,3),(155,155,155),np.uint8); cv2.rectangle(image,(110,55),(420,250),(90,135,185),-1)
        self._assert_overlaps(self._run(image,"synthetic_colour"),(110,55,421,251),.20)

    def test_clean_surface_is_bounded(self):
        image=np.full((300,500,3),155,np.uint8)
        self.assertLessEqual(len(self._run(image,"synthetic_clean").proposals),2)

    def test_lighting_gradient_is_bounded(self):
        gradient=np.tile(np.linspace(80,210,500,dtype=np.uint8),(300,1)); image=cv2.cvtColor(gradient,cv2.COLOR_GRAY2BGR)
        self.assertLessEqual(len(self._run(image,"synthetic_gradient").proposals),4)


if __name__ == "__main__":
    unittest.main()
