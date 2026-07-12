from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import pandas as pd

from experiment_tracking import METHOD_NAMES
from registered_experiment import (
    AUTOMATIC_REVIEW_STATUS, AutomaticResult, RegisteredExperimentStore, execute_plan,
    load_ground_truth, load_plan, mask_iou, match_proposals, method_summary, selected_images,
)
from research_dataset import DatasetRegistry, register_synthetic_benchmark


class MatchingTests(unittest.TestCase):
    def test_iou_matching_top_k_and_clean_image(self):
        truth=np.zeros((40,40),np.uint8); truth[10:20,10:20]=255; hit=np.zeros_like(truth); hit[12:20,12:20]=255
        metrics=match_proposals([np.zeros_like(truth),hit],truth,.1,.2)
        self.assertAlmostEqual(mask_iou(hit,truth),.64); self.assertEqual(metrics["first_true_anomaly_proposal_rank"],2); self.assertFalse(metrics["top_1_hit"]); self.assertTrue(metrics["top_3_hit"])
        clean=match_proposals([hit],np.zeros_like(truth)); self.assertIsNone(clean["top_1_hit"]); self.assertEqual(clean["false_positive_proposals"],1); self.assertIsNone(clean["proposal_recall"])
        clean_rows=pd.DataFrame([{"method":"refined contextual method","run_status":"completed","proposal_recall":None,"top_1_hit":None,"top_3_hit":None,"top_5_hit":None,"top_8_hit":None,"proposal_precision":0.,"false_positive_proposals":1,"processing_time_seconds":.1}]); summary=method_summary(clean_rows); self.assertTrue(np.isnan(summary.proposal_recall.iloc[0])); self.assertEqual(summary.proposal_precision.iloc[0],0.)

    def test_thin_crack_centroid_fallback(self):
        truth=np.zeros((50,50),np.uint8); cv2.line(truth,(5,25),(45,25),255,1); proposal=np.zeros_like(truth); proposal[20:31,20:31]=255
        metrics=match_proposals([proposal],truth,.9,.9,True); self.assertTrue(metrics["top_1_hit"])


class RegisteredExecutionSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(); cls.registry=DatasetRegistry(Path(cls.temp.name)/"research_data"); register_synthetic_benchmark(cls.registry,seed=31)
        images=cls.registry.images("synthetic-controlled").head(3)
        with cls.registry.connect() as con:
            for image_id in images.image_id: con.execute("UPDATE images SET split='test' WHERE image_id=?",(image_id,))
        cls.registry.create_experiment_plan("SYN-TEST-001","synthetic-controlled","1.0","test",3,"Development / Test","MB-01",METHOD_NAMES,{"maximum_regions":8},17)
        with cls.registry.connect() as con: cls.plan_id=con.execute("SELECT plan_id FROM experiment_plans WHERE experiment_id='SYN-TEST-001'").fetchone()[0]
        cls.store=RegisteredExperimentStore(Path(cls.temp.name)/"automatic.sqlite3"); cls.results=execute_plan(cls.registry,cls.store,cls.plan_id,1,.1,.25,"resume")
    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_loading_selected_images_and_exact_masks(self):
        plan=load_plan(self.registry,self.plan_id); images=selected_images(self.registry,plan); self.assertEqual(len(images),3)
        self.assertTrue(all(np.any(load_ground_truth(row)) or row.synthetic_anomaly_type in {"normal_texture","illumination_gradient","black_border","specular_highlights","blur","gaussian_noise"} for _,row in images.iterrows()))

    def test_four_methods_create_twelve_automatic_rows(self):
        self.assertEqual(len(self.results),12); self.assertEqual(set(self.results.method),set(METHOD_NAMES)); self.assertEqual(set(self.results.review_status),{AUTOMATIC_REVIEW_STATUS}); self.assertTrue((self.results.run_status=="completed").all())

    def test_resume_and_duplicate_run_protection(self):
        resumed=execute_plan(self.registry,self.store,self.plan_id,1,mode="resume"); self.assertEqual(len(resumed),12)
        with self.assertRaises(ValueError): execute_plan(self.registry,self.store,self.plan_id,1,mode="cancel")

    def test_method_aggregation_and_reproducibility(self):
        summary=method_summary(self.results); self.assertEqual(len(summary),4); self.assertIn("top_8_proposal_recall",summary)
        columns=["image_id","method","first_true_anomaly_proposal_rank"]; before=self.results[columns].sort_values(columns[:2]); after=execute_plan(self.registry,self.store,self.plan_id,1,mode="overwrite")[columns].sort_values(columns[:2])
        self.assertEqual(before.fillna(-1).to_records(index=False).tolist(),after.fillna(-1).to_records(index=False).tolist())

    def test_failed_pair_retry_and_plan_preserving_deletion(self):
        row=self.results.iloc[0]; failed=AutomaticResult(**{**{key:row[key] for key in AutomaticResult.__annotations__},"run_status":"failed","error_message":"forced"})
        self.store.save(failed,overwrite=True); retried=execute_plan(self.registry,self.store,self.plan_id,1,mode="retry_failed"); self.assertEqual(retried[retried.image_id==row.image_id].loc[lambda frame:frame.method==row.method].run_status.iloc[0],"completed")
        self.assertEqual(self.store.delete_results(self.plan_id,1),12); self.assertEqual(load_plan(self.registry,self.plan_id)["experiment_id"],"SYN-TEST-001")


if __name__=="__main__":unittest.main()
