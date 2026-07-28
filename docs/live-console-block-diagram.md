# Live Inspection Console Block Diagrams

The diagrams describe presentation and data flow only. They do not redefine
the frozen detector.

## Input–processing–output pipeline

```mermaid
flowchart TD
    A["INPUT IMAGE<br/>PNG / JPEG / TIFF"] --> B["VALIDATION AND NORMALISATION<br/><code>src/structvision/inputs.py</code><br/><code>src/structvision/demonstration.py</code>"]
    B --> C["PREPROCESSING<br/><code>preprocess.py</code>"]
    C --> D["VISUAL EVIDENCE EXTRACTION<br/><code>feature_extraction.py</code>"]
    D --> E["CANDIDATE REGION GENERATION / BINARY MASKS<br/><code>region_proposal.py</code>"]
    E --> F["CONTEXTUAL SCORING AND RANKING<br/><code>scoring.py</code>"]
    F --> G["TYPED RESULTS AND PROVENANCE<br/><code>src/structvision/classical.py</code><br/><code>src/structvision/types.py</code>"]
    G --> H["EXPLICIT OUTPUT FILES<br/>OVERLAY / MASKS / CSV / JSON / TECHNICAL SUMMARY"]
    I["Existing general CLI<br/><code>src/structvision/cli.py</code>"] --> B
    J["Live console wrapper<br/><code>src/structvision/live_console.py</code>"] --> B
    G --> I
    G --> J
```

## Client and method architecture

```mermaid
flowchart TD
    A["Live console client<br/>stable live demonstration"] --> C["Public StructVision API"]
    B["StructVision Streamlit client<br/>presentation alternative"] --> C
    C --> D["Frozen classical baseline<br/><b>stable default</b>"]
    D --> E["Typed in-memory results"]
    E --> F["Explicit outputs<br/>only when requested"]
    C -. "optional research branch" .-> P["PatchCore<br/><b>protected development baseline</b>"]
    C -. "optional research branch" .-> H["Proposal-guided hybrid<br/><b>rejected development candidate</b>"]
    P -. "comparison only" .-> E
    H -. "never recommended default" .-> E
```

Status language is deliberate: PatchCore remains development-only and the
hybrid remains rejected under the predeclared protocol. Neither branch is
required for the live console demonstration.
