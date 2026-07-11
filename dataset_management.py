"""Safe lifecycle operations for registered research datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil
from uuid import uuid4


@dataclass(frozen=True)
class CleanupPreview:
    dataset_id:str; versions:list[str]; files:list[str]; total_bytes:int; database_rows:int
    linked_experiments:list[str]; final_experiments:list[str]; affected_manifests:list[str]


class DatasetManager:
    def __init__(self,registry):
        self.registry=registry; self.root=registry.root.resolve(); self.trash=self.root/".trash"; self.trash.mkdir(parents=True,exist_ok=True)

    def safe_path(self,path:Path)->Path:
        candidate=Path(path)
        if ".." in candidate.parts: raise ValueError("Path traversal is not allowed")
        candidate=(candidate if candidate.is_absolute() else self.root/candidate).resolve()
        if candidate==self.root or self.root not in candidate.parents: raise ValueError("Dataset deletion is restricted to research_data/")
        return candidate

    def linked(self,dataset_id,versions=None):
        query="SELECT experiment_id,status,dataset_version FROM experiment_plans WHERE dataset_id=?"; params=[dataset_id]
        with self.registry.connect() as con: rows=con.execute(query,params).fetchall()
        if versions: rows=[row for row in rows if row["dataset_version"] in versions]
        return [dict(row) for row in rows]

    def preview(self,dataset_id,version=None,mode="complete"):
        versions=[version] if version else self.registry.datasets().query("dataset_id == @dataset_id").dataset_version.tolist()
        images=self.registry.images(dataset_id); images=images[images.dataset_version.isin(versions)] if not images.empty else images
        paths=[]
        if mode=="complete" and version:
            paths.extend(self.root/"raw"/dataset_id/name for name in images.stored_filename.tolist())
            paths.extend(Path(name) for name in images.annotation_path.tolist() if name)
        elif mode=="complete":
            paths.extend(self.root/name/dataset_id for name in ("raw","annotations","processed","splits","reports"))
        elif mode=="generated": paths.extend(self.root/name/dataset_id for name in ("processed","splits","reports"))
        elif mode in {"processed","splits","reports"}: paths.append(self.root/mode/dataset_id)
        files=[]
        for path in paths:
            path=self.safe_path(path)
            if path.is_file(): files.append(path)
            elif path.exists(): files.extend(item for item in path.rglob("*") if item.is_file())
        links=self.linked(dataset_id,versions)
        return CleanupPreview(dataset_id,versions,[str(p) for p in files],sum(p.stat().st_size for p in files),len(images)+len(versions),[r["experiment_id"] for r in links],[r["experiment_id"] for r in links if r["status"]=="Final Research Evaluation"],[str(self.registry.registry_dir/"dataset_manifest.json")])

    def delete(self,dataset_id,version=None,mode="complete",reviewer="",unlink=False,cascade_development=False):
        preview=self.preview(dataset_id,version,mode)
        if preview.final_experiments and mode in {"complete","metadata"}: raise ValueError("Final research experiments can never be deleted or silently unlinked")
        if preview.linked_experiments and mode in {"complete","metadata"} and not (unlink or cascade_development): raise ValueError("Dataset is linked to saved experiments")
        stamp=datetime.now().strftime("%Y%m%dT%H%M%S%f"); destination=self.trash/stamp/dataset_id; moved=[]
        for filename in preview.files:
            source=self.safe_path(Path(filename)); relative=source.relative_to(self.root); target=self.safe_path(destination/relative)
            target.parent.mkdir(parents=True,exist_ok=True); shutil.move(str(source),str(target)); moved.append({"source":str(source),"trash":str(target)})
        datasets=self.registry.datasets(); datasets=datasets[(datasets.dataset_id==dataset_id)&(datasets.dataset_version.isin(preview.versions))]
        images=self.registry.images(dataset_id); images=images[images.dataset_version.isin(preview.versions)] if not images.empty else images
        removed=0
        if mode in {"metadata","complete"}:
            clause="dataset_id=?"+(" AND dataset_version=?" if version else ""); params=(dataset_id,version) if version else (dataset_id,)
            with self.registry.connect() as con:
                removed+=con.execute(f"DELETE FROM images WHERE {clause}",params).rowcount
                removed+=con.execute(f"DELETE FROM datasets WHERE {clause}",params).rowcount
                if cascade_development: removed+=con.execute("DELETE FROM experiment_plans WHERE dataset_id=? AND status != 'Final Research Evaluation'",(dataset_id,)).rowcount
                elif unlink: con.execute("UPDATE experiment_plans SET dataset_id='[deleted dataset]' WHERE dataset_id=?",(dataset_id,))
        audit={"audit_id":str(uuid4()),"dataset_id":dataset_id,"versions":preview.versions,"reviewer":reviewer,"timestamp":datetime.now().isoformat(),"deletion_mode":mode,"files_moved":moved,"records_removed":removed,"dataset_rows":datasets.to_dict("records"),"image_rows":images.to_dict("records")}
        audit_path=destination/"deletion_audit.json"; audit_path.parent.mkdir(parents=True,exist_ok=True); audit_path.write_text(json.dumps(audit,indent=2)); self.registry._write_manifest_json()
        return {"deleted_files":len(moved),"deleted_records":removed,"trash_path":str(destination),"preserved_links":[] if unlink or cascade_development else preview.linked_experiments}

    def restore(self,audit_path):
        audit_path=self.safe_path(Path(audit_path)); audit=json.loads(audit_path.read_text()); restored=0
        for item in audit["files_moved"]:
            source=self.safe_path(Path(item["trash"])); target=self.safe_path(Path(item["source"])); target.parent.mkdir(parents=True,exist_ok=True)
            if source.exists(): shutil.move(str(source),str(target)); restored+=1
        if audit["deletion_mode"] in {"metadata","complete"}:
            with self.registry.connect() as con:
                for row in audit.get("dataset_rows",[]):
                    metadata={key:row[key] for key in row if key!="registered_timestamp"}
                    con.execute("INSERT OR REPLACE INTO datasets VALUES(?,?,?,?)",(row["dataset_id"],row["dataset_version"],json.dumps(metadata),row["registered_timestamp"]))
                for row in audit.get("image_rows",[]):
                    columns=list(row); con.execute(f"INSERT OR REPLACE INTO images ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",[row[key] for key in columns])
            self.registry._write_manifest_json()
        return restored

    def empty_trash(self):
        count=sum(1 for path in self.trash.rglob("*") if path.is_file())
        for child in list(self.trash.iterdir()):
            safe=self.safe_path(child); shutil.rmtree(safe) if safe.is_dir() else safe.unlink()
        return count

    def set_split_eligible(self,image_id,eligible):
        with self.registry.connect() as con: con.execute("UPDATE images SET split_eligible=?, split=? WHERE image_id=?",(int(eligible),"unassigned",image_id))
        self.registry._write_manifest_json()

    def delete_item(self,image_id,delete_file=False,delete_annotation=False):
        images=self.registry.images(); row=images[images.image_id==image_id]
        if row.empty: raise KeyError(image_id)
        item=row.iloc[0]; moved=[]
        for path in ([self.root/"raw"/item.dataset_id/item.stored_filename] if delete_file else [])+([Path(item.annotation_path)] if delete_annotation and item.annotation_path else []):
            source=self.safe_path(path)
            if source.exists(): target=self.safe_path(self.trash/datetime.now().strftime("%Y%m%dT%H%M%S%f")/item.dataset_id/source.relative_to(self.root)); target.parent.mkdir(parents=True,exist_ok=True); shutil.move(source,target); moved.append(str(target))
        if delete_file:
            with self.registry.connect() as con: con.execute("DELETE FROM images WHERE image_id=?",(image_id,))
        elif delete_annotation:
            with self.registry.connect() as con: con.execute("UPDATE images SET annotation_path='' WHERE image_id=?",(image_id,))
        self.registry._write_manifest_json(); return moved
