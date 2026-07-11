from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest
from uuid import uuid4

import cv2
import numpy as np
import pandas as pd

from experiment_tracking import (
    DuplicateRecordError,
    ExperimentRecord,
    ExperimentStore,
    METHOD_NAMES,
    build_experiment_records,
    delete_legacy_records,
    experiment_tables,
    legacy_record_indices,
    migrate_legacy_records,
    with_version,
)
from feature_extraction import extract_feature_maps
from labeling import build_annotation
from region_proposal import propose_regions
from research_evaluation import (
    EFFICIENCY_CHART_COLUMNS,
    FALSE_PROPOSAL_COLUMN,
    OUTCOME_CHART_COLUMNS,
    RECALL_CHART_COLUMNS,
)


def make_record(
    experiment_id="EXP-A", image="a.png", method="refined contextual method",
    status="Final Research Evaluation", review_status="fully_reviewed",
    accepted=1, rejected=1, uncertain=0, not_reviewed=0,
    outcome="anomaly present", ground_truth="verified ground truth", rank=1, version=1,
):
    eligible = outcome == "anomaly present" and ground_truth != "unknown" and rank is not None
    return ExperimentRecord(
        record_id=str(uuid4()), experiment_id=experiment_id, experiment_version=version,
        reviewer_id="REV-1", experiment_status=status, image_filename=image, method=method,
        recorded_timestamp="2026-07-11T10:00:00", review_status=review_status,
        final_proposals=8, accepted=accepted, rejected=rejected, uncertain=uncertain,
        not_reviewed=not_reviewed, image_outcome=outcome, dataset_source="test",
        image_provenance="synthetic", license_status="internal", ground_truth_status=ground_truth,
        ground_truth_recall_override=False,
        development_notes="", review_start_time="2026-07-11T09:59:00",
        review_completion_time="2026-07-11T10:00:00",
        review_duration_seconds=60.0 if review_status != "not_reviewed" else None,
        first_accepted_true_anomaly_rank=rank,
        true_anomaly_found_top_1=(rank <= 1) if eligible else None,
        true_anomaly_found_top_3=(rank <= 3) if eligible else None,
        true_anomaly_found_top_5=(rank <= 5) if eligible else None,
        true_anomaly_found_top_8=(rank <= 8) if eligible else None,
        proposals_reviewed_before_first_useful=rank if eligible else None,
    )


class ExperimentStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.store=ExperimentStore(Path(self.temp.name)/"experiments.sqlite3")

    def tearDown(self):
        self.temp.cleanup()

    def test_delete_one_row_preserves_unrelated(self):
        rows=[make_record(),make_record("EXP-B","b.png")]
        self.store.save(rows)
        deleted,ids=self.store.delete_record_ids([rows[0].record_id])
        self.assertEqual(deleted,1); self.assertEqual(ids,["EXP-A"])
        self.assertEqual(self.store.dataframe().iloc[0]["experiment_id"],"EXP-B")

    def test_delete_full_experiment_id(self):
        rows=[make_record(method=METHOD_NAMES[0]),make_record(method=METHOD_NAMES[1]),make_record("EXP-B","b.png")]
        self.store.save(rows); deleted,ids=self.store.delete_where("experiment_id","EXP-A")
        self.assertEqual(deleted,2); self.assertEqual(ids,["EXP-A"]); self.assertEqual(len(self.store.dataframe()),1)

    def test_delete_by_image_filename(self):
        rows=[make_record(),make_record("EXP-B","a.png")]
        self.store.save(rows); deleted,ids=self.store.delete_where("image_filename","a.png")
        self.assertEqual(deleted,2); self.assertEqual(set(ids),{"EXP-A","EXP-B"})

    def test_clear_development_preserves_final(self):
        development=make_record(status="Development / Test")
        final=make_record("EXP-B","b.png")
        self.store.save([development,final]); deleted,_=self.store.delete_where("experiment_status","Development / Test")
        remaining=self.store.dataframe()
        self.assertEqual(deleted,1); self.assertEqual(remaining.iloc[0]["experiment_status"],"Final Research Evaluation")

    def test_reset_requires_confirmation(self):
        self.store.save([make_record()])
        with self.assertRaises(PermissionError): self.store.reset()
        self.assertEqual(len(self.store.dataframe()),1)
        deleted,_=self.store.reset(confirmed=True)
        self.assertEqual(deleted,1); self.assertTrue(self.store.dataframe().empty)

    def test_duplicate_prevention_overwrite_and_new_version(self):
        row=make_record(); self.store.save([row])
        with self.assertRaises(DuplicateRecordError): self.store.save([replace(row,record_id=str(uuid4()))])
        replacement=replace(row,record_id=str(uuid4()),development_notes="overwritten")
        self.store.save([replacement],duplicate_action="overwrite")
        self.assertEqual(self.store.dataframe().iloc[0]["development_notes"],"overwritten")
        versioned=with_version([replacement],self.store.next_version("EXP-A"))
        self.store.save(versioned)
        self.assertEqual(set(self.store.dataframe()["experiment_version"]),{1,2})

    def test_recalculation_after_deletion(self):
        first=make_record(rank=1); second=make_record("EXP-B","b.png",rank=5)
        self.store.save([first,second])
        _,before=experiment_tables(self.store.dataframe())
        self.assertEqual(before.iloc[0]["top_1_proposal_recall"],.5)
        self.store.delete_record_ids([second.record_id])
        _,after=experiment_tables(self.store.dataframe())
        self.assertEqual(after.iloc[0]["top_1_proposal_recall"],1.0)


class MetricSemanticsTests(unittest.TestCase):
    def test_not_reviewed_baseline_semantics(self):
        image=np.full((180,300,3),155,np.uint8); cv2.rectangle(image,(80,55),(220,130),(70,130,195),-1)
        features=extract_feature_maps(image); result=propose_regions(image,features,"semantics",min_area=20,max_regions=3)
        annotations=[build_annotation("x.png",proposal,index==0,"other_surface_anomaly",decision="accept" if index==0 else "reject") for index,proposal in enumerate(result.proposals)]
        records=build_experiment_records("EXP","REV","x.png","anomaly present","2026-01-01T00:00:00","2026-01-01T00:01:00",annotations,result,features,
            experiment_status="Final Research Evaluation",ground_truth_status="verified ground truth")
        for baseline in records[:-1]:
            self.assertEqual(baseline.review_status,"not_reviewed"); self.assertIsNone(baseline.accepted)
            self.assertIsNone(baseline.rejected); self.assertIsNone(baseline.uncertain)
            self.assertEqual(baseline.not_reviewed,baseline.final_proposals)

    def test_acceptance_rate_denominator_excludes_uncertain(self):
        row=make_record(accepted=2,rejected=1,uncertain=5)
        _,summary=experiment_tables([row])
        self.assertAlmostEqual(summary.iloc[0]["annotation_acceptance_rate"],2/3)

    def test_uncertain_and_unknown_are_excluded_from_recall(self):
        uncertain=make_record(outcome="uncertain",rank=None)
        unknown=make_record("EXP-B","b.png",ground_truth="unknown",rank=1)
        _,summary=experiment_tables([uncertain,unknown])
        self.assertEqual(summary.iloc[0]["eligible_images"],0)
        self.assertTrue(pd.isna(summary.iloc[0]["top_1_proposal_recall"]))

    def test_unknown_ground_truth_requires_explicit_override(self):
        overridden=replace(make_record(ground_truth="unknown",rank=1),ground_truth_recall_override=True,
            true_anomaly_found_top_1=True,true_anomaly_found_top_3=True,true_anomaly_found_top_5=True,true_anomaly_found_top_8=True)
        _,summary=experiment_tables([overridden])
        self.assertEqual(summary.iloc[0]["eligible_images"],1)
        self.assertEqual(summary.iloc[0]["top_1_proposal_recall"],1.0)

    def test_undefined_metrics_are_na_not_zero(self):
        baseline=make_record(method=METHOD_NAMES[0],review_status="not_reviewed",accepted=None,rejected=None,uncertain=None,not_reviewed=8,outcome="no anomaly",rank=None)
        _,summary=experiment_tables([baseline])
        self.assertTrue(pd.isna(summary.iloc[0]["annotation_acceptance_rate"]))
        self.assertTrue(pd.isna(summary.iloc[0]["mean_review_time_seconds"]))
        self.assertTrue(pd.isna(summary.iloc[0]["mean_proposals_reviewed_before_first_useful"]))

    def test_development_final_filtering(self):
        development=make_record(status="Development / Test")
        final=make_record("EXP-B","b.png")
        final_table,_=experiment_tables([development,final])
        both,_=experiment_tables([development,final],include_development=True)
        development_table,_=experiment_tables([development,final],include_development=True,development_only=True)
        self.assertEqual(len(final_table),1); self.assertEqual(len(both),2); self.assertEqual(len(development_table),1)
        self.assertEqual(development_table.iloc[0]["experiment_status"],"Development / Test")

    def test_chart_column_mapping(self):
        self.assertEqual(list(RECALL_CHART_COLUMNS.values()),["top_1_proposal_recall","top_3_proposal_recall","top_5_proposal_recall","top_8_proposal_recall"])
        self.assertEqual(set(OUTCOME_CHART_COLUMNS.values()),{"mean_accepted_proposals_per_image","mean_rejected_proposals_per_image","mean_uncertain_proposals_per_image","mean_not_reviewed_proposals_per_image"})
        self.assertEqual(FALSE_PROPOSAL_COLUMN,"mean_false_proposals_per_image")
        self.assertEqual(len(EFFICIENCY_CHART_COLUMNS),3)


class LegacyMigrationTests(unittest.TestCase):
    def test_legacy_detection_migration_and_selected_deletion(self):
        rows=[{"experiment_id":"OLD","reviewer_id":"R","image_filename":"old.png","method":"contour-only baseline","final_proposals":4},
              {"record_id":"modern","review_status":"fully_reviewed","not_reviewed":0,"experiment_status":"Final Research Evaluation"}]
        self.assertEqual(legacy_record_indices(rows),[0])
        with tempfile.TemporaryDirectory() as directory:
            store=ExperimentStore(Path(directory)/"db.sqlite3")
            self.assertEqual(migrate_legacy_records(rows,[0],store),1)
            migrated=store.dataframe().iloc[0]
            self.assertEqual(migrated["review_status"],"not_reviewed"); self.assertEqual(migrated["not_reviewed"],4)
            path=Path(directory)/"legacy.json"; path.write_text(json.dumps(rows),encoding="utf-8")
            self.assertEqual(delete_legacy_records(path,rows,[0]),1)
            self.assertEqual(len(json.loads(path.read_text())),1)


if __name__=="__main__": unittest.main()
