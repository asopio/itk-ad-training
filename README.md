# ITK AD Training
A package for training PyTorch models for anomaly detection for QC of ATLAS ITk PCBs.

Dependencies:
- PyTorch
- Lightning
- Comet (optional, for tracking logs on the web)
- [defect_detection](https://github.com/lovaslin/defect_detection/tree/main) (autoencoder model developed by Louis Vaslin)

## Usage
Training can be configured using config files. The base config can be found in `configs/default_config.yaml` 

```
python cli.py fit --config <config name>
```

To log training information using Comet, create and account on https://www.comet.com, obtain an API key, and set the following environment variable
```
export COMET_API_KEY="YOUR-API-KEY"
```