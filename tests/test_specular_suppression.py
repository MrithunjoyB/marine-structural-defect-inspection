import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace

import cv2
import numpy as np

from ablation_study import CONFIG_BY_ID,RERANK_ONLY,configuration_snapshot
from feature_extraction import extract_feature_maps
from region_proposal import _coherence_metrics,_region_metrics,_specular_evidence,propose_regions
from registered_experiment import AutomaticResult,RegisteredExperimentStore


def evidence(kind):
    rng=np.random.default_rng(19); base=np.clip(140+rng.normal(0,5,(120,160,1)),0,255).astype(np.uint8); image=np.repeat(base,3,axis=2); mask=np.zeros((120,160),np.uint8)
    if kind=="highlight":cv2.circle(image,(80,60),12,(255,255,255),-1);cv2.circle(mask,(80,60),12,255,-1)
    elif kind=="crack":cv2.line(image,(20,100),(140,20),(245,245,245),3);cv2.line(mask,(20,100),(140,20),255,5)
    elif kind=="pitting":
        for centre in ((60,60),(80,55),(100,65)):cv2.circle(image,centre,7,(245,245,245),-1);cv2.circle(mask,centre,8,255,-1)
    else:image[40:80,50:110]=rng.integers(20,100,(40,60,3),dtype=np.uint8);mask[40:80,50:110]=255
    maps=extract_feature_maps(image); x,y,w,h=cv2.boundingRect(mask); metrics=_region_metrics(image,maps,mask,(x,y,x+w,y+h));metrics.update(_coherence_metrics(mask,maps.anomaly_strength));return image,mask,_specular_evidence(image,mask,(x,y,x+w,y+h),metrics)


class SpecularEvidenceTests(unittest.TestCase):
    def test_deterministic_bounded_finite_score(self):
        _,_,first=evidence("highlight");_,_,second=evidence("highlight");self.assertEqual(first,second);self.assertGreaterEqual(first["specular_likelihood"],0);self.assertLessEqual(first["specular_likelihood"],1);self.assertTrue(np.isfinite(first["specular_likelihood"]))
    def test_smooth_low_chroma_highlight_scores_high(self):self.assertGreater(evidence("highlight")[2]["specular_likelihood"],.8)
    def test_dark_textured_anomaly_scores_low(self):self.assertLess(evidence("dark")[2]["specular_likelihood"],.3)
    def test_crack_safeguard_is_conservative(self):
        data=evidence("crack")[2];self.assertGreater(data["crack_safeguard"],.8);self.assertLess(data["specular_likelihood"]*(1-.9*data["crack_safeguard"]),.2)
    def test_pitting_safeguard_is_conservative(self):self.assertGreaterEqual(evidence("pitting")[2]["pitting_safeguard"],.5)
    def test_configuration_and_snapshot_persistence(self):
        definition=CONFIG_BY_ID["ABL-RERANK-SPECULAR-SUPPRESS"];self.assertTrue(definition.config.specular_suppression);self.assertFalse(CONFIG_BY_ID["ABL-RERANK-ONLY"].config.specular_suppression);snapshot=configuration_snapshot(definition,"E",1,42,"manifest",{"iou":.1,"mask_overlap":.25});self.assertTrue(snapshot["enabled_components"]["specular_suppression"]);self.assertEqual(snapshot["enabled_components"]["specular_rejection_threshold"],.50)
    def test_diagnostic_serialization(self):json.dumps(evidence("highlight")[2],allow_nan=False)
    def test_pipeline_diagnostic_serialization_and_rejection_reason(self):
        image,_,_=evidence("highlight");result=propose_regions(image,extract_feature_maps(image),"specular_diagnostic",min_area=20,ablation=CONFIG_BY_ID["ABL-RERANK-SPECULAR-SUPPRESS"].config);payload=result.diagnostics.to_dict();json.dumps(payload,allow_nan=False);self.assertTrue(payload["Specular diagnostics"]);self.assertTrue(all("before_priority_score" in item and "after_priority_score" in item for item in payload["Specular diagnostics"]));self.assertTrue(any(item["decision"] in {"penalized","rejected"} for item in payload["Specular diagnostics"]))
    def test_stored_result_schema_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            store=RegisteredExperimentStore(Path(directory)/"results.sqlite3");self.assertEqual(list(store.dataframe().columns),list(AutomaticResult.__annotations__))
    def test_disabled_suppression_preserves_existing_configuration(self):
        self.assertEqual(RERANK_ONLY,replace(RERANK_ONLY,specular_suppression=False));image,_,_=evidence("highlight");maps=extract_feature_maps(image);a=propose_regions(image,maps,"specular_disabled_a",min_area=20,ablation=RERANK_ONLY);b=propose_regions(image,maps,"specular_disabled_b",min_area=20,ablation=replace(RERANK_ONLY,specular_penalty_weight=.1,specular_rejection_threshold=.2));self.assertEqual([p.bbox for p in a.proposals],[p.bbox for p in b.proposals]);self.assertEqual([p.priority.score for p in a.proposals],[p.priority.score for p in b.proposals])


if __name__=="__main__":unittest.main()
