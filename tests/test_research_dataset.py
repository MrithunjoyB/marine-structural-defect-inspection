from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import pandas as pd

from research_dataset import (
    DatasetMetadata,DatasetRegistry,create_configuration_snapshot,hamming_distance,ingest_files,
    licence_allows_public_export,prepare_split,register_synthetic_benchmark,sha256_file,
    validate_annotation,
)
from research_evaluation import chart_height,has_valid_metric
from dataset_management import DatasetManager
from synthetic_benchmark import generate_cases


def metadata(dataset_id="dataset-a",ground="verified dataset annotation",annotation="none",licence="CC-BY-4.0"):
    return DatasetMetadata(dataset_id,"Dataset A","1.0","public research dataset","Source","https://example.test","Author",licence,True,True,"Citation","2026-07-12","structural inspection",ground,annotation,"")


def png_bytes(value=120,size=(48,64)):
    image=np.full((size[0],size[1],3),value,np.uint8); ok,data=cv2.imencode(".png",image); assert ok; return data.tobytes()


def _write_temp(root,name,data):
    path=Path(root)/name; path.write_bytes(data); return path


class ResearchDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.registry=DatasetRegistry(Path(self.temp.name)/"research_data")
    def tearDown(self):self.temp.cleanup()

    def test_dataset_registration_and_required_metadata(self):
        self.registry.register_dataset(metadata()); self.assertEqual(len(self.registry.datasets()),1)
        with self.assertRaises(ValueError):self.registry.register_dataset(replace(metadata("bad"),dataset_name=""))

    def test_hash_duplicate_corrupt_and_manifest(self):
        content=png_bytes(); files=[("a.png",content),("renamed.png",content),("broken.png",b"not image"),("empty.png",b"")]
        records,report=ingest_files(self.registry,metadata(),files)
        self.assertEqual(records[1].duplicate_status,"exact duplicate"); self.assertEqual(report.exact_duplicates,1)
        self.assertEqual(report.corrupted_files,2); self.assertTrue((self.registry.registry_dir/"dataset_manifest.json").exists())
        path=Path(self.temp.name)/"hash.png"; path.write_bytes(content); self.assertEqual(sha256_file(path),records[0].sha256_hash)

    def test_annotation_bounds_validation(self):
        self.assertTrue(validate_annotation("YOLO bounding boxes",b"0 0.5 0.5 0.2 0.2",100,100)[0])
        self.assertFalse(validate_annotation("YOLO bounding boxes",b"0 1.4 0.5 0.2 0.2",100,100)[0])
        empty=np.zeros((20,20),np.uint8); _,data=cv2.imencode(".png",empty); self.assertFalse(validate_annotation("binary masks",data.tobytes(),20,20)[0])

    def test_deterministic_split_and_duplicate_leakage_prevention(self):
        files=[(f"image_{i}.png",png_bytes(80+i*12)) for i in range(8)]
        ingest_files(self.registry,metadata(),files)
        first,leaks,_=prepare_split(self.registry,"dataset-a",seed=17)
        first_map=dict(zip(first.image_id,first.split)); second,leaks2,_=prepare_split(self.registry,"dataset-a",seed=17)
        self.assertEqual(first_map,dict(zip(second.image_id,second.split))); self.assertFalse(any(leaks.values())); self.assertFalse(any(leaks2.values()))

    def test_group_leakage_prevention(self):
        records,_=ingest_files(self.registry,metadata(),[(f"g{i}.png",png_bytes(90+i*20)) for i in range(6)])
        with self.registry.connect() as con:
            con.execute("UPDATE images SET group_id='component-1' WHERE image_id IN (?,?)",(records[0].image_id,records[1].image_id))
        result,leaks,_=prepare_split(self.registry,"dataset-a",seed=9)
        grouped=result[result.group_id=="component-1"]; self.assertEqual(grouped.split.nunique(),1); self.assertEqual(leaks["group_across_splits"],0)

    def test_licence_warning_and_override(self):
        unknown=metadata(licence="unknown")
        self.assertFalse(licence_allows_public_export(unknown)[0]); self.assertTrue(licence_allows_public_export(unknown,True)[0])

    def test_unknown_ground_truth_blocks_final_experiment(self):
        ingest_files(self.registry,metadata(ground="unknown"),[("a.png",png_bytes())])
        with self.assertRaises(ValueError):self.registry.create_experiment_plan("EXP","dataset-a","1.0","all",1,"Final Research Evaluation","REV",["refined contextual method"],{},42)
        path=self.registry.create_experiment_plan("EXP","dataset-a","1.0","all",1,"Development / Test","REV",["refined contextual method"],{},42)
        self.assertTrue(path.exists())

    def test_configuration_snapshot(self):
        snapshot=create_configuration_snapshot({"preprocessing":{"clahe":True},"border_margin":.02,"maximum_regions":8})
        for key in ("preprocessing_settings","code_commit_hash","python_version","package_versions","operating_system"):self.assertIn(key,snapshot)

    def test_synthetic_benchmark_registration(self):
        records,report=register_synthetic_benchmark(self.registry,seed=5)
        self.assertGreaterEqual(len(records),11); self.assertEqual(report.valid_images,len(records))
        self.assertTrue(all(record.annotation_path for record in records))

    def test_annotations_are_not_image_duplicates(self):
        image=png_bytes(); annotations={"a.png":png_bytes(255),"b.png":png_bytes(255)}
        records,_=ingest_files(self.registry,metadata(annotation="binary masks"),[("a.png",image),("b.png",png_bytes(121))],annotations)
        self.assertNotEqual(records[0].sha256_hash,records[1].sha256_hash)
        self.assertNotEqual(records[1].duplicate_type,"exact image duplicate")

    def test_synthetic_hashes_seeds_and_reproducibility(self):
        first,params=generate_cases(27,with_parameters=True); second,_=generate_cases(27,with_parameters=True)
        encode=lambda image:cv2.imencode(".png",image)[1].tobytes()
        hashes=[sha256_file(_write_temp(self.temp.name,name,encode(image))) for name,(image,_) in first.items()]
        self.assertEqual(len(hashes),len(set(hashes))); self.assertEqual(list(first),list(second))
        self.assertTrue(all(np.array_equal(first[name][0],second[name][0]) for name in first)); self.assertEqual(len({p["derived_seed"] for p in params.values()}),len(params))

    def test_exact_duplicates_are_excluded_and_near_variants_share_split(self):
        content=png_bytes(100); records,_=ingest_files(self.registry,metadata(),[("a.png",content),("copy.png",content),("b.png",png_bytes(101)),("c.png",png_bytes(180))])
        self.assertEqual(records[1].split_eligible,0); result,_,_=prepare_split(self.registry,"dataset-a",seed=3)
        self.assertEqual(result.loc[result.image_id==records[1].image_id,"split"].iloc[0],"unassigned")
        near=result[result.image_id.isin([records[0].image_id,records[2].image_id])]; self.assertEqual(near.split.nunique(),1)

    def test_deletion_modes_restore_and_unrelated_preservation(self):
        ingest_files(self.registry,metadata(),[("a.png",png_bytes())]); ingest_files(self.registry,metadata("other"),[("b.png",png_bytes(180))])
        manager=DatasetManager(self.registry); preview=manager.preview("dataset-a",None,"complete"); self.assertGreater(preview.database_rows,0)
        result=manager.delete("dataset-a",mode="complete",reviewer="tester"); self.assertEqual(len(self.registry.images("dataset-a")),0); self.assertEqual(len(self.registry.images("other")),1)
        audit=next(Path(result["trash_path"]).glob("deletion_audit.json")); manager.restore(audit); self.assertEqual(len(self.registry.images("dataset-a")),1)
        manager.delete("dataset-a",mode="metadata",reviewer="tester"); self.assertEqual(len(self.registry.images("dataset-a")),0)

    def test_generated_only_and_path_guards(self):
        ingest_files(self.registry,metadata(),[("a.png",png_bytes())]); generated=self.registry.root/"reports"/"dataset-a"/"extra.json"; generated.write_text("{}")
        DatasetManager(self.registry).delete("dataset-a",mode="generated",reviewer="tester"); self.assertFalse(generated.exists()); self.assertEqual(len(self.registry.images("dataset-a")),1)
        manager=DatasetManager(self.registry)
        with self.assertRaises(ValueError):manager.safe_path(Path("../README.md"))
        outside=Path(self.temp.name)/"outside"; outside.mkdir(); link=self.registry.root/"raw"/"escape"; link.symlink_to(outside,target_is_directory=True)
        with self.assertRaises(ValueError):manager.safe_path(link/"file.png")

    def test_linked_experiment_protection(self):
        ingest_files(self.registry,metadata(),[("a.png",png_bytes())]); self.registry.create_experiment_plan("EXP","dataset-a","1.0","all",1,"Development / Test","REV",["refined contextual method"],{},1)
        manager=DatasetManager(self.registry)
        with self.assertRaises(ValueError):manager.delete("dataset-a",mode="complete")

    def test_chart_empty_suppression_and_adaptive_sizing(self):
        empty=pd.DataFrame({"metric":[np.nan]}); self.assertFalse(has_valid_metric(empty,["metric"])); self.assertFalse(has_valid_metric(pd.DataFrame({"metric":[0]}),["metric"],True))
        self.assertLess(chart_height(1),chart_height(2)); self.assertLess(chart_height(4),chart_height(5))

    def test_no_deprecated_streamlit_width_usage(self):
        root=Path(__file__).resolve().parents[1]
        deprecated="use_container_"+"width"
        offenders=[str(path) for path in root.rglob("*.py") if "venv" not in path.parts and deprecated in path.read_text(encoding="utf-8")]
        self.assertEqual(offenders,[])


if __name__=="__main__":unittest.main()
