# ITK AD Training
A package for training PyTorch models for anomaly detection for QC of ATLAS ITk PCBs.

Dependencies:
- PyTorch
- Lightning
- Comet (optional, for tracking logs on the web)
- defect_detection (autoencoder model developed by Louis Vaslin)

## Usage
Training can be configured using config files. The base config can be found in `configs/default_config.yaml` 

```
python cli.py fit --config <config name>
```