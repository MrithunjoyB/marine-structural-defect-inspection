# Private Dataset Adapter Contract

## Status

This is an interface and secure-intake specification only. No private collaborator data, private path, credential, storage integration, or adapter implementation is included.

## Boundary

```python
from collections.abc import Iterable, Mapping
from typing import Protocol

class PrivateDatasetAdapter(Protocol):
    def samples(self) -> Iterable["SampleRecord"]: ...
    def open_image(self, sample: "SampleRecord") -> "ImageData": ...
    def ground_truth(self, sample: "SampleRecord") -> "GroundTruth | None": ...
    def groups(self, sample: "SampleRecord") -> Mapping[str, str]: ...
```

The adapter belongs outside detector logic and outside the public repository when it contains private storage references. It converts authorised private records to typed inputs; the frozen detectors remain unchanged.

## Required sample record

| Field | Contract |
|---|---|
| `sample_id` | Immutable, non-identifying unique ID |
| `image_content_sha256` | SHA-256 of approved encoded image content |
| `source_reference` | Opaque private-storage reference outside the public repository |
| `colour_space` | Explicit `GRAY`, `RGB`, `BGR`, `RGBA`, or `BGRA` |
| `alpha_handling` | Explicit when alpha exists |
| `width`, `height`, `bit_depth` | Decoded acquisition properties |
| `vessel_id` | Pseudonymous vessel/group identifier |
| `component_id` | Pseudonymous component/structure identifier |
| `session_id` | Acquisition-session identifier |
| `camera_id` | Camera/sensor identifier |
| `acquisition_group_id` | Group used to prevent leakage |
| `acquisition_timestamp` | Approved precision and timezone, or omitted |
| `anomaly_status` | `unknown`, `no_anomaly`, or `anomaly_present` |
| `annotation_type` | none, image label, box, polygon, or binary mask |
| `reviewer_id` | Pseudonymous reviewer identity |
| `annotation_version` | Immutable version plus supersession link |
| `licence_status` | Written permitted-use state |
| `confidentiality_classification` | Agreed handling class |
| `role` | train, development, or test; locked before method access |
| `split_lock_hash` | Hash over IDs, content, groups, roles, and policy |

Unknown values remain explicit; they are not replaced with a plausible default.

## `ImageData`

`open_image` returns in-memory encoded bytes or a validated `uint8` array plus declared colour/alpha semantics. It must:

- verify the content hash before returning;
- reject unsupported bit depth/channel layout;
- avoid logging pixels or private storage locations;
- avoid creating a public-repository copy;
- avoid cloud upload unless a separate written protocol authorises it;
- release buffers according to the approved retention policy.

## Ground truth

`ground_truth` returns `None` when labels are unavailable. When present it includes:

- annotation type and coordinate convention;
- image-content identity;
- annotation content hash;
- reviewer pseudonym and role;
- annotation version and timestamp;
- review/adjudication status;
- uncertainty or “cannot assess” semantics;
- superseded-version linkage;
- licence/confidentiality status.

An empty anomaly mask is not interchangeable with missing ground truth.

## Grouping and split lock

`groups` supplies every known leakage unit: vessel, component, session, camera, acquisition group, burst/video, source export, and near-duplicate family. Split construction occurs outside detector code, before development/test access, and keeps linked groups in one role.

The split-lock hash covers:

```text
policy version
ordered sample IDs
image hashes
ground-truth hashes or explicit missing values
all grouping fields
role assignment
exclusions and reasons
```

Role changes create a new prospective protocol; they never rewrite a prior result.

## Isolation requirements

The adapter must remain isolated from:

- `src/structvision/classical.py` and every detector implementation;
- Git-tracked private paths or metadata;
- public documentation containing private identifiers;
- automatic registry mutation;
- automatic cloud upload, telemetry, or analytics;
- evaluation-store writes without an explicit approved sink;
- train/development/test reassignment after lock.

## Secure intake checklist

1. Confirm data owner, controller, custodian, and technical contact.
2. Obtain written licence and permitted research/competition/publication uses.
3. Agree confidentiality class, storage region, encryption, access list, retention, deletion, and incident process.
4. Identify personally, commercially, operationally, or security-sensitive metadata.
5. Define pseudonymisation and the separately controlled re-identification map.
6. Inventory formats, bit depths, resolutions, colour profiles, alpha, compression, cameras, and sessions.
7. Define anomaly ontology without forcing uncertain samples into a positive/negative class.
8. Define annotation type, tool, reviewer expertise, adjudication, and versioning.
9. Hash original content before conversion; retain conversion provenance.
10. Define acquisition groups and exact/near-duplicate audit.
11. Lock roles and compute split-lock identity before method access.
12. Approve a small normal-operation intake smoke test separately from performance evaluation.
13. Approve a prospective pilot specification and explicit result sink.
14. Review any intended public examples individually; default to private.
15. Document secure deletion and archive obligations at pilot close.

## Proposed connection sequence

```mermaid
flowchart LR
    A["Written authority and data map"] --> B["Private adapter implementation"]
    B --> C["Content and metadata validation"]
    C --> D["Group-aware split lock"]
    D --> E["Normal-operation smoke checks"]
    E --> F["Predeclared prospective pilot"]
    F --> G["Explicit private result store"]
    G --> H["Joint technical review"]
```

No stage requires changing the detector core.
