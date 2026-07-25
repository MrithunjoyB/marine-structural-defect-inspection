# StructVision-AI Pseudocode

This pseudocode follows the implementation order and preserves role, threshold, artifact, provenance, and proposal-order semantics.

## 1. Frozen classical analysis

Implementation: `src/structvision/api.py::StructuralAnomalyDetector.analyse`, `src/structvision/inputs.py::normalise_input`, `src/structvision/classical.py::run_frozen_classical`.

```text
FROZEN_CLASSICAL_ANALYSE(image, image_id, colour_space, alpha_handling, metadata):
    REQUIRE image_id is non-empty
    normalised ← normalise_input(image, colour_space, alpha_handling)
    REQUIRE normalised is contiguous uint8 BGR

    ENTER one process-wide legacy lock
    CREATE one temporary artifact directory
    IMPORT protected preprocess, feature_extraction, scoring, region_proposal modules
    HASH protected module sources
    REDIRECT protected feature/mask/output paths to the temporary directory

    processed ← apply_preprocessing(normalised.BGR, exact DetectorConfig.preprocessing)
    feature_maps ← extract_feature_maps(processed, exact DetectorConfig.features)
    legacy_result ← propose_regions(
        processed,
        feature_maps,
        exact ProposalConfig arguments,
        exact AblationConfig switches
    )

    FOR legacy_proposal IN legacy_result.proposals IN returned order:
        final_mask ← read exact returned final-mask file as uint8
        raw_mask ← read exact returned raw-mask file as uint8
        rank ← integer suffix of protected region ID
        bbox ← protected half-open box; Proposal verifies it against final_mask
        proposal ← immutable Proposal with protected scores and diagnostics

    provenance ← build_provenance(actual source hashes, temporary artifact count)
    RESTORE protected path globals
    DELETE temporary directory before return
    RETURN immutable AnalysisResult with unchanged proposal order
```

The demonstration invokes this through `src/structvision/demonstration.py::analyse_demonstration_image` with no sink.

## 2. PatchCore normal-only fitting

Implementation: `src/structvision/normal_feature/patchcore.py::NormalFeatureAnomalyDetector.fit_normal`.

```text
PATCHCORE_FIT_NORMAL(samples, normal_fit_manifest_hash):
    REQUIRE ordered unique samples
    REQUIRE every sample role = normal_fit and outcome = no_anomaly
    REQUIRE each supplied image hash matches image content
    REQUIRE normal_fit_manifest_hash is SHA-256
    VERIFY exact dependency versions
    VERIFY local official weight SHA-256
    FORCE offline backend policy, deterministic CPU, seed 42, one thread
    BUILD official PatchcoreModel with pre_trained = false
    STRICTLY LOAD verified safetensors checkpoint into official backbone

    FOR sample IN samples IN manifest order:
        prepared ← aspect-preserving 256×416 letterbox with recorded geometry
        embedding_store.APPEND(official PatchCore embedding(prepared.tensor))

    complete_memory ← concatenate embeddings in input order
    RESET deterministic seed
    coreset_indices ← official KCenterGreedy(complete_memory, ratio = 0.001)
    memory_bank ← complete_memory[coreset_indices] as float32
    RECORD configuration, inputs, hashes, coreset, preprocessing, environment,
           hardware, Git state, creation time
    CREATE content-addressed NormalFeatureModelArtifact
    IF and only if artifact_sink was explicitly supplied: artifact_sink.write(artifact)
    RETURN artifact
```

The professor demonstration never calls fitting.

## 3. PatchCore image analysis

Implementation: `src/structvision/normal_feature/patchcore.py::NormalFeatureAnomalyDetector.score` and `.analyse`; `src/structvision/normal_feature/proposal_extraction.py::extract_proposals`.

```text
PATCHCORE_ANALYSE(image, model_artifact, calibration_artifact, operating_point_id):
    VERIFY model configuration and official-weight identities
    VERIFY calibration.model_artifact_hash = model.artifact_hash
    VERIFY calibration extraction-policy hash = config.proposal hash
    operating_point ← calibration.operating_point(operating_point_id)

    prepared ← normalise input and record deterministic letterbox geometry
    model ← reconstruct official PatchCore model from verified weight + immutable memory bank
    raw_output ← model(prepared.tensor) under inference mode
    full_map ← inverse-project raw anomaly map to original analysed-image coordinates
    image_score ← raw PatchCore pred_score distance

    proposals ← extract_proposals(
        full_map,
        threshold = operating_point.threshold,
        connectivity = 8,
        morphology = none,
        minimum area = 16,
        maximum count = 8
    )
    ORDER proposals by descending component distance, then deterministic component identity
    RETURN NormalFeatureAnalysisResult with map/artifact/configuration hashes
```

The professor client fixes `operating_point_id = fp-budget-0.50` and does not calibrate.

## 4. Candidate-level hybrid fusion

Implementation: `src/structvision/hybrid/detector.py::ProposalGuidedHybridDetector.analyse`, `src/structvision/hybrid/features.py::candidate_evidence` and `normalised_components`.

```text
HYBRID_ANALYSE(image, frozen_classical, hybrid_model, fusion_artifact, budget = 0.50):
    VERIFY fusion selection_status = selected
    VERIFY classical configuration hash, hybrid model artifact hash,
           and environment-lock hash match fusion artifact
    classical_result ← frozen_classical.analyse(image)
    patchcore_score ← normal_feature_detector.score(image, hybrid_model)
    REQUIRE input hashes and image coordinate shapes are identical
    operating_point ← fusion_artifact.operating_point(budget)
    REQUIRE budget is one of frozen 0.25, 0.50, 1.00

    FOR proposal IN classical_result.proposals IN frozen classical order:
        raw_features ← [
            classical priority,
            classical evidence,
            classical heuristic reliability,
            PatchCore mask-interior mean,
            PatchCore mask-interior q95,
            PatchCore high-support fraction,
            PatchCore context contrast,
            PatchCore local spatial agreement
        ]
        scaled_features ← artifact q05/q95 affine scaling, clamped to [0,1]
        classical_component ← mean(first 3 scaled features)
        normality_component ← mean(last 5 scaled features)
        hybrid_score ← 0.60 × classical_component + 0.40 × normality_component
        selected ← hybrid_score ≥ frozen operating_point.threshold
        RETAIN complete diagnostic and unchanged classical mask

    ordered_selected ← sort selected by (-hybrid_score, proposal_id)
    ASSIGN contiguous one-based final ranks
    RETURN HybridAnalysisResult(selected proposals, all pre-threshold diagnostics)
```

The development artifact selected a fusion configuration on `hybrid_fusion_fit`; the later one-shot holdout decision classified it as a **rejected development candidate**. Runtime replay does not change that research status.

## 5. V2 experiment execution

Implementation: `src/structvision/executor.py::ExperimentExecutorV2.execute`; learned extension `src/structvision/learned_executor.py::DevelopmentExperimentExecutorV2.execute`; hybrid extension `src/structvision/hybrid/experiment.py::HybridDevelopmentExperimentExecutorV2.execute`.

```text
V2_EXECUTE(specification, ordered_samples, optional_sink):
    VERIFY canonical specification hash
    VERIFY exact structvision-eval-v2 policy/version/hash/thresholds
    REQUIRE ordered sample IDs exactly equal immutable selected-image IDs
    VERIFY every image and ground-truth content hash before analysis
    VERIFY each method's executable configuration against specification

    FOR sample IN selected order:
        FOR method IN specification method order:
            RUN the already-defined public detector
            VERIFY protected provenance
            CONVERT returned masks to ProposalSet without reranking
            READ and align ground truth
            EVALUATE with fixed one-to-one v2 matching and metrics
            APPEND one in-memory completed row, or a typed failed row when fail_fast = false

    BUILD one complete execution summary with expected/attempted/completed/failed counts
    IF and only if optional_sink was supplied: optional_sink.write(report)
    RETURN in-memory report
```

The professor demonstration does not invoke this procedure and cannot write evaluation rows.

## 6. Future professor-data ingestion

Specification boundary: `docs/professor-data-adapter.md`. No implementation or professor data is present.

```text
PROFESSOR_DATA_INGEST(adapter, approved_intake_manifest):
    REQUIRE written licence, usage authority, confidentiality class, and storage owner
    REQUIRE source references remain outside the public repository
    samples ← adapter.samples() in immutable sample-ID order

    FOR sample IN samples:
        image ← adapter.open_image(sample)
        VERIFY content hash, colour declaration, resolution, and acquisition metadata
        truth ← adapter.ground_truth(sample) or None
        VERIFY annotation type, reviewer pseudonym, annotation version, and status
        groups ← adapter.groups(sample)
        REQUIRE vessel/component/session/camera/acquisition grouping is explicit
        RECORD only pseudonymous, approved metadata in a private registry

    LOCK train/development/test roles before method access
    COMPUTE split-lock hash over IDs, content hashes, roles, and groups
    AUDIT exact/near-duplicate and group leakage
    REQUIRE independent approval before any prospective evaluation
    PASS typed records to public detector/evaluation adapters; never modify detector core
```
