# 🧘🏽 ViTPose Plugin
This plugin essentially makes it easy to add human pose estimation capabilities to any FiftyOne dataset that already has person detections.

### Plugin Overview
This plugin integrates ViTPose (a Vision Transformer-based pose estimation model) into FiftyOne, allowing users to detect human body keypoints in images that already have person bounding boxes detected.

* Supports multiple ViTPose architectures

* Hardware acceleration (CUDA/MPS) when available

* Confidence threshold filtering

* Delegation support for distributed processing

* Handles normalized coordinate conversions

* Integrates with FiftyOne's Dataset and UI

### Note: This plugin makes assumptions.

##### It assumes you have bounding box detections on your dataset already.

If you do not have bounding boxes on your Dataset already, you can add them quite easily, for example:

```python
from ultralytics import YOLO

yolo_world_model = YOLO("yolov8x-worldv2.pt")  # or choose yolov8m/l-world.pt

# Define custom classes
yolo_world_model.set_classes(["person"])

dataset.apply_model(
    yolo_world_model,
    label_field="bbox_detections"
)
```
##### It assumes you already have metadata computed on your dataset.

If you don't already have metadata computed, you can do so as follows:

```python
dataset.compute_metadata()
```


#### Supported Models

This plugin supports all [currently available checkpoints](https://huggingface.co/collections/usyd-community/vitpose-677fcfd0a0b2b5c8f79c4335) (as of Jan 2025).

- `usyd-community/vitpose-plus-small`
- `usyd-community/vitpose-base`
- `usyd-community/vitpose-base-simple`
- `usyd-community/vitpose-plus-base`
- `usyd-community/vitpose-plus-large`
- `usyd-community/vitpose-plus-huge`
- `usyd-community/vitpose-base-coco-aic-mpii`

## Installation

If you haven't already, install FiftyOne:

```shell
pip install -U fiftyone transformers
```

Then, install the plugin:

```shell
fiftyone plugins download https://github.com/harpreetsahota204/vitpose-plugin
```

## Usage in FiftyOne App

You can estimate poses using ViTPose directly through the FiftyOne App:

1. Launch the FiftyOne App with your dataset
2. Open the "Operators Browser" by clicking on the Operator Browser icon above the sample grid or by typing backtick (`)
3. Type "VitPose Human Pose Estimator"
4. Configure the following parameters:
   - **ViTPose Model**: Select one of the supported models
   - **Bounding Box Field**: The field where your bounding box detections are
   - **Output Field**: Enter the name for the field to store key points
   - **Confidence Threshold**: Confidence threshold for keypoint detection
   
5. Click "Execute" to compute keypoints for your dataset


## Operators

### `vitpose_keypoint_estimator`

Estimate keypoints for human pose estimation with VitPose

## Operator usage via SDK

Once the plugin has been installed, you can instantiate the operator as follows:

```python
import fiftyone.operators as foo

vitpose_operator = foo.get_operator("@harpreetsahota/vitpose/vitpose_keypoint_estimator")

```
You can then estimate poses on your dataset by running the operator with your desired parameters:

```python
# Run the operator on your dataset
vitpose_operator(
    dataset,
    model_name="usyd-community/vitpose-plus-small",  # Select from one of the supported models
    bbox_field="bbox_field", # Name of the field where your bounding box detections are stored.
    output_field="vitpose_estimates",  # Name of the field to store the Keypoints in.
    confidence_threshold= 0.55 #Confidence threshold for keypoint detection
)
```

If you're running in a notebook, it's recommended to launch a [Delegated operation](https://docs.voxel51.com/plugins/using_plugins.html#delegated-operations) by running `fiftyone delegated launch` in terminal, then run as follows:

```python
await vitpose_operator(
    dataset,
    model_name="usyd-community/vitpose-plus-small",
    bbox_field="bbox_field", 
    output_field="vitpose_estimates",
    confidence_threshold= 0.55
)
```
# Citation

You can read the paper here.

```bibtex
@article{xu2022vitposesimplevisiontransformer,
  title={ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation},
  author={Yufei Xu and Jing Zhang and Qiming Zhang and Dacheng Tao},
  year={2022},
  eprint={2204.12484},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2204.12484}
}

```