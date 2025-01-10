import os

from fiftyone.core.utils import add_sys_path
import fiftyone.operators as foo
from fiftyone.operators import types

VITPOSE_ARCHS = {
    "vitpose-base":"usyd-community/vitpose-base",
    "vitpose-plus-base":"usyd-community/vitpose-plus-base",
    "vitpose-base-simple":"usyd-community/vitpose-base-simple",
    "vitpose-base-coco-aic-mpii":"usyd-community/vitpose-base-coco-aic-mpii"
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

            icon="/assets/icons8-meta.svg",

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
            description=(
                "Name of the field to store the Keypoints in."
                " NOTE: This model assumes your dataset already has bounding box detections."
                )
            )

        inputs.str(
            "output_field",            
            required=True,
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