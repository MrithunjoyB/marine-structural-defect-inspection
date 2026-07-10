from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from experiment_tracking import (
    METHOD_NAMES,
    build_experiment_records,
    experiment_tables,
    load_experiment_records,
    save_experiment_records,
)
from feature_extraction import extract_feature_maps
from labeling import build_annotation
from region_proposal import propose_regions


class ExperimentTrackingTests(unittest.TestCase):
    def _fixture(self):
        image=np.full((220,360,3),155,np.uint8)
        cv2.rectangle(image,(100,70),(260,150),(70,125,195),-1)
        features=extract_feature_maps(image)
        result=propose_regions(image,features,"experiment_fixture",min_area=20,max_regions=8)
        self.assertTrue(result.proposals)
        annotations=[]
        for index,proposal in enumerate(result.proposals):
            decision="accept" if index==0 else ("reject" if index%2 else "uncertain")
            annotations.append(build_annotation("fixture.png",proposal,decision=="accept","other_surface_anomaly",decision=decision))
        return features,result,annotations

    def test_records_capture_timing_outcomes_and_top_k(self):
        features,result,annotations=self._fixture()
        records=build_experiment_records(
            "EXP-001","REV-01","fixture.png","anomaly present",
            "2026-07-11T10:00:00","2026-07-11T10:02:00",
            annotations,result,features,
        )
        self.assertEqual(tuple(record.method for record in records),METHOD_NAMES)
        refined=records[-1]
        self.assertEqual(refined.review_duration_seconds,120)
        self.assertEqual(refined.first_accepted_true_anomaly_rank,1)
        self.assertTrue(refined.true_anomaly_found_top_1)
        self.assertEqual(refined.accepted,1)
        self.assertEqual(refined.accepted+refined.rejected+refined.uncertain,len(result.proposals))

    def test_dataset_summary_and_exports(self):
        features,result,annotations=self._fixture()
        records=build_experiment_records(
            "EXP-002","REV-02","fixture.png","anomaly present",
            "2026-07-11T10:00:00","2026-07-11T10:01:00",
            annotations,result,features,
        )
        image_table,summary=experiment_tables(records)
        self.assertEqual(len(image_table),4); self.assertEqual(len(summary),4)
        self.assertIn("top_8_proposal_recall",summary.columns)
        self.assertIn("mean_review_time_seconds",summary.columns)
        with tempfile.TemporaryDirectory() as directory:
            csv_path=Path(directory)/"results.csv"; json_path=Path(directory)/"results.json"
            save_experiment_records(records,csv_path,json_path)
            self.assertTrue(csv_path.exists()); self.assertEqual(len(load_experiment_records(json_path)),4)

    def test_ids_and_outcome_are_required(self):
        features,result,annotations=self._fixture()
        with self.assertRaises(ValueError):
            build_experiment_records("","REV","fixture.png","anomaly present","2026-01-01T00:00:00","2026-01-01T00:00:01",annotations,result,features)


if __name__=="__main__": unittest.main()
