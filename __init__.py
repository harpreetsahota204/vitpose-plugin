import os
import fiftyone as fo
from fiftyone.core.utils import add_sys_path
import fiftyone.operators as foo
from fiftyone.operators import types

VITPOSE_ARCHS = {
    "vitpose-base": "usyd-community/vitpose-base",
    "vitpose-plus-small": "usyd-community/vitpose-plus-small",
    "vitpose-plus-base": "usyd-community/vitpose-plus-base",
    "vitpose-plus-large": "usyd-community/vitpose-plus-large",
    "vitpose-plus-huge": "usyd-community/vitpose-plus-huge",
    "vitpose-base-simple": "usyd-community/vitpose-base-simple",
    "vitpose-base-coco-aic-mpii": "usyd-community/vitpose-base-coco-aic-mpii"
}

with add_sys_path(os.path.dirname(os.path.abspath(__file__))):
    from pose_estimator import (
        run_pose_estimation
    )

def _handle_calling(
        uri, 
        sample_collection, 
        model_name,
        bbox_field,
        output_field,
        confidence_threshold,
        delegate=False
        ):
    ctx = dict(dataset=sample_collection)

    params = dict(
        model_name=model_name,
        bbox_field=bbox_field,
        output_field=output_field,
        confidence_threshold=confidence_threshold,
        delegate=delegate
        )
    return foo.execute_operator(uri, ctx, params=params)

class ViTPoseEstimator(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            # The operator's URI: f"{plugin_name}/{name}"
            name="vitpose_keypoint_estimator",  # required

            # The display name of the operator
            label="ViTPose Human Pose Estimator",  # required

            # A description for the operator
            description="Estimate keypoints for human pose estimation with VitPose",

            icon="/assets/yoga-position-svgrepo-com.svg",

            )

    def resolve_input(self, ctx):
        """Implement this method to collect user inputs as parameters
        that are stored in `ctx.params`.

        Returns:
            a `types.Property` defining the form's components
        """
        inputs = types.Object()

        model_dropdown = types.Dropdown(label="Choose the ViTPose model you want to use:")
        
        for arch_key, arch_value in VITPOSE_ARCHS.items():
            model_dropdown.add_choice(arch_value, label=arch_key)

        inputs.enum(
            "model_name",
            values=model_dropdown.values(),
            label="ViTPose Model",
            description="Select from one of the supported models. Note: The model weights will be downloaded from Hugging Face.",
            view=model_dropdown,
            required=True
        )

        inputs.str(
            "bbox_field",            
            required=True,
            label="Bounding Box Field",
            description=(
                "Name of the field where your bounding box detections are stored."
                " NOTE: This model assumes your dataset already has bounding box detections."
                )
            )

        inputs.str(
            "output_field",            
            required=True,
            label="Output Field",
            description="Name of the field to store the Keypoints in."
            )
        
        inputs.bool(
            "delegate",
            default=False,
            required=True,
            label="Delegate execution?",
            description=("If you choose to delegate this operation you must first have a delegated service running. "
            "You can launch a delegated service by running `fiftyone delegated launch` in your terminal"),
            view=types.CheckboxView(),
        )

        inputs.float(
            "confidence_threshold",
            label="Confidence Threshold",
            default=0.35,
            min=0.0,
            max=1.0,
            float=True,
            description="Confidence threshold for keypoint detection",
        )

        inputs.view_target(ctx)

        return types.Property(inputs)

    def resolve_delegation(self, ctx):
        """Implement this method if you want to programmatically *force*
        this operation to be delegated or executed immediately.

        Returns:
            whether the operation should be delegated (True), run
            immediately (False), or None to defer to
            `resolve_execution_options()` to specify the available options
        """
        return ctx.params.get("delegate", False)


    def execute(self, ctx):
        """Executes the actual operation based on the hydrated `ctx`.
        All operators must implement this method.

        This method can optionally be implemented as `async`.

        Returns:
            an optional dict of results values
        """
        view = ctx.target_view()
        model_name = ctx.params.get("model_name")
        bbox_field = ctx.params.get("bbox_field")
        output_field = ctx.params.get("output_field")
        confidence_threshold = ctx.params.get("confidence_threshold")
        
        run_pose_estimation(
            dataset=view,
            model_name = model_name,
            bbox_field = bbox_field,
            output_field = output_field,
            confidence_threshold = confidence_threshold
            )
        
        keypoint_labels = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        # Define skeleton edges
        edges = [
            [0, 1], [0, 2], [1, 3], [2, 4],  # Face
            [5, 7], [7, 9], [6, 8], [8, 10],  # Arms
            [5, 6], [5, 11], [6, 12], [11, 12],  # Torso
            [11, 13], [13, 15], [12, 14], [14, 16]  # Legs
        ]

        skeleton = fo.KeypointSkeleton(labels=keypoint_labels, edges=edges)

        view.skeletons = {
            output_field: skeleton,
            }


        
        ctx.ops.reload_dataset()

    def __call__(
            self, 
            sample_collection, 
            model_name,
            bbox_field,
            output_field,
            confidence_threshold,
            delegate=False
            ):
        return _handle_calling(
            self.uri,
            sample_collection,
            model_name,
            bbox_field,
            output_field,
            confidence_threshold,
            delegate=delegate
            )

def register(p):
    """Always implement this method and register() each operator that your
    plugin defines.
    """
    p.register(ViTPoseEstimator)