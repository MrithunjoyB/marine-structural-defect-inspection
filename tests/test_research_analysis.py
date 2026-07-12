from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ablation_study import ABLATION_CONFIGS,CONFIG_BY_ID,configuration_snapshot,contribution_table,validate_comparison
from region_proposal import AblationConfig
from research_analysis import bootstrap_ci,category_summary,filter_results,filtered_csv,filtered_json,paired_advanced,paired_bootstrap,positive_negative_summary,win_tie_loss


def rows():
    data=[]
    for image_id,filename,category,outcome in (("i1","thin_crack_01.png","thin_crack","anomaly_present"),("i2","pitting_cluster_01.png","pitting_cluster","anomaly_present"),("i3","normal_texture_01.png","normal_texture","no_anomaly"),("i4","specular_highlights_01.png","specular_highlights","no_anomaly")):
        for method in ("contour-only baseline","fixed-threshold baseline","multi-scale fused method","refined contextual method"):
            contextual=method=="refined contextual method"; fused=method=="multi-scale fused method"; positive=outcome=="anomaly_present"; matched=positive and (contextual or fused)
            data.append({"result_id":f"{image_id}-{method}","experiment_id":"SYN-BALANCED-001","experiment_version":1,"plan_id":"p1","dataset_id":"synthetic-controlled","dataset_version":"1.0","dataset_split":"test","image_id":image_id,"image_filename":filename,"anomaly_type":category,"clean_artefact_type":category if not positive else "","image_outcome":outcome,"method":method,"review_status":"automatically_evaluated","run_status":"completed","final_proposals":2,"first_true_anomaly_proposal_rank":1 if matched else np.nan,"top_1_hit":True if matched else (False if positive else None),"top_3_hit":True if matched else (False if positive else None),"top_5_hit":True if matched else (False if positive else None),"top_8_hit":True if matched else (False if positive else None),"proposal_precision":.5 if matched else 0.,"proposal_recall":1. if matched else (0. if positive else np.nan),"mean_iou":.4 if matched else 0.,"best_iou":.6 if matched else 0.,"false_positive_proposals":1 if matched else 2,"false_negative_anomalies":0 if matched or not positive else 1,"processing_time_seconds":.2 if contextual else .1,"recorded_timestamp":"2026-01-01"})
    return pd.DataFrame(data)


class ResultBrowserTests(unittest.TestCase):
    def setUp(self):self.frame=rows()
    def test_search_and_combined_filters(self):
        self.assertEqual(len(filter_results(self.frame,search="balanced")),16); self.assertEqual(len(filter_results(self.frame,search="THIN_crack")),4)
        filtered=filter_results(self.frame,filters={"experiment_id":["SYN-BALANCED-001"],"method":["refined contextual method"],"anomaly_type":["thin_crack"]}); self.assertEqual(len(filtered),1)
    def test_image_method_category_and_numeric_filters(self):
        self.assertEqual(len(filter_results(self.frame,filters={"image_filename":["thin_crack_01.png"]})),4); self.assertEqual(len(filter_results(self.frame,quick="Compare advanced methods")),8)
        self.assertTrue((filter_results(self.frame,numeric={"proposal_precision":(.4,None)}).proposal_precision>=.4).all())
    def test_empty_reset_equivalent_and_exports(self):
        self.assertTrue(filter_results(self.frame,search="missing").empty); self.assertEqual(len(filter_results(self.frame)),len(self.frame)); self.assertIn(b"SYN-BALANCED-001",filtered_csv(self.frame)); self.assertIn(b'"experiment_id"',filtered_json(self.frame))
    def test_selected_image_four_method_comparison(self):self.assertEqual(filter_results(self.frame,quick="Compare all methods for selected image",selected_image="thin_crack_01.png").method.nunique(),4)


class CategoryAndPairTests(unittest.TestCase):
    def setUp(self):self.frame=rows()
    def test_category_aggregation_and_denominators(self):
        summary=category_summary(self.frame); self.assertEqual(set(summary.category),{"thin_crack","pitting_cluster","normal_texture","specular_highlights"}); clean=summary[summary.category=="normal_texture"]; self.assertTrue(clean.proposal_recall.isna().all()); self.assertEqual(clean.mean_false_proposals_per_image.mean(),2.)
        self.assertEqual(len(positive_negative_summary(self.frame)),8)
    def test_pairing_differences_and_outcomes(self):
        paired=paired_advanced(self.frame); self.assertEqual(len(paired),4); self.assertTrue(paired.difference_processing_time_seconds.notna().all()); counts=win_tie_loss(paired); self.assertEqual(counts["images"],4); self.assertEqual(win_tie_loss(paired,"thin_crack")["images"],1)
    def test_missing_category(self):self.assertFalse("future_category" in set(category_summary(self.frame).category))


class AblationAndStatisticsTests(unittest.TestCase):
    def test_stable_unique_ids_and_default_unchanged(self):
        ids=[item.configuration_id for item in ABLATION_CONFIGS]; self.assertEqual(len(ids),len(set(ids))); self.assertEqual(CONFIG_BY_ID["ABL-FULL"].config,AblationConfig()); self.assertFalse(CONFIG_BY_ID["ABL-NO-BORDER"].config.border_penalty)
    def test_reproducible_snapshot_and_comparison_guard(self):
        first=configuration_snapshot(ABLATION_CONFIGS[0],"E",1,42,"hash",{"iou":.1}); second=configuration_snapshot(ABLATION_CONFIGS[0],"E",1,42,"hash",{"iou":.1}); self.assertEqual(first["snapshot_hash"],second["snapshot_hash"]); self.assertTrue(validate_comparison([first,second])); bad={**second,"random_seed":7}
        with self.assertRaises(ValueError):validate_comparison([first,bad])
    def test_bootstrap_determinism_and_small_sample(self):
        self.assertEqual(bootstrap_ci([1,2,3],200,4),bootstrap_ci([1,2,3],200,4)); self.assertTrue(np.isnan(bootstrap_ci([1],100,1)[0])); paired=paired_advanced(rows()); self.assertEqual(paired_bootstrap(paired,"proposal_precision",200,3),paired_bootstrap(paired,"proposal_precision",200,3))


if __name__=="__main__":unittest.main()
