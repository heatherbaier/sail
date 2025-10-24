# SAIL: The Spatial AI Library

**SAIL** (Spatial AI Library) is an open-source framework for building, training, and explaining spatial deep learning models — designed to make GeoAI accessible to both experts and newcomers.

It provides a modular configuration-based workflow for:
- Training models on satellite imagery and geospatial data  
- Evaluating and validating spatial predictions  
- Running explainability and perturbation analyses (e.g., SIMBA, SIA)  
- Integrating new models, samplers, and datasets with minimal code

---

## Quick Start

**Install directly from GitHub:**
```bash
pip install git+https://github.com/heatherbaier/sail.git
```

**Run your first experiment:**

python launch.py --config configs/phl_geoconv_regression.yaml


This example trains a GeoConv-based regression model on imagery and tabular data defined in the configuration file.

All outputs — models, logs, and metrics — will be saved to the specified output_dir in your config.

**Configuration-Based Design**

Every experiment in SAIL is defined through a YAML configuration file that specifies:

- Dataset source and parameters
- Model architecture and initialization
- Training hyperparameters
- Validation and explainability settings

Example:

```yaml
task: train
experiment_name: "usaid_dhs_pct_water_improved"
output_dir: "artifacts/checkpoints/phl_geoconv_income"

dataset:
  type: json
  data_root: "/data/phl"
  prefix: "pct_improved_water_africa"
  batch_size: 16
  img_size: [224, 224]

model:
  name: geoconv
  params:
    pretrained: false

trainer:
  epochs: 1
  lr: 1e-4
  device: "cuda"
```


***Documentation***

Full usage instructions, tutorials, and developer guides are available in the SAIL Wiki: https://github.com/heatherbaier/sail/wiki

There, you'll find:
- How to structure and run experiments
- How to register new models or datasets
- How to extend SAIL with explainability or spatial robustness modules
- Examples and notebooks for GeoAI research and teaching