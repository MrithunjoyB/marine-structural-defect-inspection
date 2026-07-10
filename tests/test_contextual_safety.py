from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import cv2
import numpy as np

from config import DEFAULT_LABEL_CLASSES
from dataset_export import export_dataset
from evaluation import evaluate_method
from feature_extraction import extract_feature_maps
from labeling import ReviewedAnnotation, build_annotation
from region_proposal import _bbox_from_mask, _split_candidate, _suppress_overlaps, _Candidate, propose_regions
from scoring import score_architecture


class ContextualSafetyTests(unittest.TestCase):
    def test_black_borders_are_suppressed(self):
        image=np.full((260,420,3),150,np.uint8); image[:30]=0; image[-30:]=0
        result=propose_regions(image,extract_feature_maps(image),"border_test",min_area=20)
        self.assertTrue(all(p.border_penalty<.5 for p in result.proposals))
        self.assertTrue(all(p.bbox[1]>=5 and p.bbox[3]<=255 for p in result.proposals))

    def test_large_incoherent_mask_splits(self):
        mask=np.zeros((240,400),np.uint8); cv2.rectangle(mask,(25,70),(375,170),255,-1)
        heat=np.zeros_like(mask); cv2.circle(heat,(90,120),35,240,-1); cv2.circle(heat,(310,120),35,250,-1)
        candidate=_Candidate(mask,(25,70,376,171))
        self.assertGreaterEqual(len(_split_candidate(candidate,heat,mask.size)),2)

    def test_contextual_colour_anomaly_beats_normal_texture(self):
        rng=np.random.default_rng(4); image=np.clip(150+rng.normal(0,8,(260,420,3)),0,255).astype(np.uint8)
        cv2.rectangle(image,(150,80),(300,200),(70,135,205),-1)
        result=propose_regions(image,extract_feature_maps(image),"context_colour",min_area=20)
        self.assertTrue(result.proposals)
        self.assertGreater(max(p.local_colour_contrast for p in result.proposals),.05)
        self.assertGreater(max(p.anomaly_evidence_score for p in result.proposals),20)

    def test_stable_normal_region_is_not_high_evidence(self):
        image=np.full((260,420,3),155,np.uint8); cv2.line(image,(0,130),(419,130),(145,145,145),10)
        result=propose_regions(image,extract_feature_maps(image),"stable_normal",min_area=20)
        self.assertLess(max((p.anomaly_evidence_score for p in result.proposals),default=0),75)

    def test_label_defaults_to_unassigned(self):
        self.assertEqual(DEFAULT_LABEL_CLASSES[0],"unassigned")
        proposal=SimpleNamespace(region_id="R001",bbox=(1,1,10,10),mask_path=Path("mask.png"),priority=SimpleNamespace(score=20,label="Low"))
        annotation=build_annotation("x.png",proposal,False,"")
        self.assertEqual(annotation.label,"unassigned"); self.assertEqual(annotation.decision,"reject")

    def test_unlabelled_accept_is_rejected_by_export(self):
        ann=ReviewedAnnotation("x.png","R001",True,"accept","unassigned",(1,1,10,10),"missing.png","refined",20,"Low","","2026-01-01T00:00:00")
        with self.assertRaises(ValueError): export_dataset(Path("missing.png"),(20,20),[ann])

    def test_evaluation_metrics(self):
        metrics=evaluate_method([(0,0,10,10)],[(0,0,10,10)],image_name="case",method="exact")
        self.assertEqual(metrics.proposal_recall_iou_050,1.0); self.assertEqual(metrics.average_best_iou,1.0)
        self.assertEqual(metrics.area_over_coverage_ratio,0.0); self.assertEqual(metrics.area_under_coverage_ratio,0.0)

    def test_pipeline_counts_and_final_masks_are_consistent(self):
        image=np.full((240,400,3),155,np.uint8); cv2.line(image,(30,180),(370,60),(35,35,35),7)
        result=propose_regions(image,extract_feature_maps(image),"pipeline_order",min_area=20,max_regions=5)
        diagnostics=result.diagnostics
        self.assertLessEqual(diagnostics.after_merging,diagnostics.after_splitting)
        self.assertLessEqual(diagnostics.after_overlap_suppression,diagnostics.after_merging)
        self.assertLessEqual(diagnostics.final_count,5)
        for proposal in result.proposals:
            mask=cv2.imread(str(proposal.mask_path),0)
            self.assertGreater(cv2.countNonZero(mask),0); self.assertEqual(_bbox_from_mask(mask),proposal.bbox)

    def test_overlapping_duplicates_do_not_survive_nms(self):
        image=np.full((160,240,3),150,np.uint8); fm=extract_feature_maps(image)
        first=np.zeros(image.shape[:2],np.uint8); first[40:110,50:160]=255
        second=np.zeros_like(first); second[45:105,55:155]=255
        kept,rejected=_suppress_overlaps([_Candidate(first,(50,40,160,110)),_Candidate(second,(55,45,155,105))],image,fm)
        self.assertEqual(len(kept),1); self.assertEqual(len(rejected),1)

    def test_contextual_evidence_dominates_stability(self):
        low_evidence=score_architecture({"local_texture_contrast":0,"local_colour_contrast":0,"local_entropy_contrast":0,"edge_concentration":0,"gradient_contrast":0,"geometric_irregularity":0},
            {"perturbation_stability":1,"connectedness":1,"boundary_smoothness":1,"scale_agreement":1,"segmentation_coherence":1},.2,0)[2]
        high_evidence=score_architecture({"local_texture_contrast":1,"local_colour_contrast":1,"local_entropy_contrast":1,"edge_concentration":1,"gradient_contrast":1,"geometric_irregularity":1},
            {"perturbation_stability":0,"connectedness":0,"boundary_smoothness":0,"scale_agreement":0,"segmentation_coherence":0},.2,0)[2]
        self.assertGreater(high_evidence,low_evidence)


if __name__=="__main__": unittest.main()
