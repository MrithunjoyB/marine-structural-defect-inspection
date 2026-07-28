# Data Card: Protected Normal-Feature Development Cohort

## Identity

- Protocol: `structvision-normal-feature-development-v1`
- Source: `synthetic-expanded` v1.0 train and validation only
- Manifest: `2aa40b9db145a37522775b7ac605ae201b91e564cde881528fd6d41f449f3d58`
- Classification: development-only — non-confirmatory

## Composition

The cohort has 91 clean `normal_fit` images and 72 `calibration_validation` images (34 clean, 38 positive) across all ten expected categories. No test role exists. Images are 500×300 deterministic synthetic PNGs; positive truth is a registered binary mask and clean truth is an implicit verified zero mask.

## Protection and provenance

The selector excludes exact hashes, conservative perceptual candidates, and declared group crossings with the pilot, historical tests, and prior verification identities. It verifies every selected encoded image and truth identity and keeps source/template/acquisition grouping. The manifest is canonical, immutable, and deterministic. See [the full protocol](development-data-protocol.md).

## Appropriate use and limitations

Appropriate use is normal-only memory construction, validation-only operating-point calibration, and non-confirmatory development analysis. It must not be used as an independent test, real-world validation, private-data proxy, or transferability evidence. Generator-family dependence, conservative candidate exclusion, missing acquisition groups, and unresolved semantic near-duplicate risk limit interpretation.
