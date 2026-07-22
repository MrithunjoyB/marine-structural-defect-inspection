# Historical Pilot/Final Dataset Overlap Audit

## Status

This is a read-only audit of the existing registry. No dataset, mask, registry row, split, plan, or result row was modified.

| Field | Value |
|---|---|
| Audit algorithm | `structvision-dataset-overlap-v1` |
| Perceptual candidate screen | `legacy-dhash-64-candidate-screen` |
| Hamming-distance threshold | 3 |
| Registry manifest SHA-256 | `bc266fcac6009c022aa45f43e97d0461f71f3b00bcfbdf1781a6fdd2eaea7ba4` |
| Left dataset | `synthetic-expanded-pilot` v1.0, 80 images |
| Right dataset | `synthetic-expanded` v1.0, 500 images |
| Evidence classification | **historical engineering comparison — not confirmatory** |

## Exact SHA-256 Findings

All 80 pilot files recur byte-for-byte in the 500-image expanded dataset. The exact overlaps occur in the expanded split as follows:

| Expanded split | Exact pilot overlaps |
|---|---:|
| Train | 51 |
| Validation | 16 |
| Test | 13 |
| **Total** | **80** |

Every one of the ten categories contributes eight exact overlaps. The 13 final-test overlaps are five `thin_crack`, five `normal_texture`, and three `illumination_gradient` images. Therefore the expanded test is not a protected confirmatory test.

## Perceptual And Group Findings

After excluding the 80 exact pairs, the legacy 64-bit difference-hash screen reports 242 additional cross-dataset candidate pairs at Hamming distance at most 3. These are candidates for review, not proof that the images are duplicates. Conversely, absence from this candidate list would not prove semantic uniqueness or exclude crop, resize, illumination, or other transformed relationships.

The audit records 400 cross-dataset source-group pair crossings and 400 template-group pair crossings. Acquisition-group IDs are not populated, so acquisition-group independence cannot be assessed. All 80 exact cross-dataset pairs were historically declared “unique” within their separate registrations; those cross-dataset uniqueness statuses are unsupported.

## Corrected Conclusion

The prior “zero near-duplicate leakage” statement was not established by the implemented perceptual-hash check. The exact pilot/final overlap alone is sufficient to invalidate a confirmatory interpretation. The current expanded results remain preserved as historical engineering evidence and must not be hidden, deleted, or relabelled in storage.

Future protocol work must create an immutable cross-dataset audit before test lock, populate acquisition/source/template groups, review perceptual candidates, and prevent pilot or development images from entering the confirmatory split.
