# Model Weights

Place the trained YOLO weight file here as:

```text
best.pt
```

The app works fully without this file using classical feature extraction, anomaly region proposal, human labeling, and dataset export.

When `best.pt` exists, trained YOLO predictions are shown separately from classical candidate regions. Do not commit large model weights unless the repository is configured for Git LFS.
