from typing import List, Tuple, Dict
from PIL import Image
import fiftyone as fo
import torch
from transformers import AutoProcessor, VitPoseForPoseEstimation

def get_device() -> str:
    """Helper function to determine the best available device.
    
    Returns:
        str: Device identifier ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon (MPS) device")
    else:
        device = "cpu"
        print("Using CPU device")
    return device

def init_model(model_name: str) -> Tuple[AutoProcessor, VitPoseForPoseEstimation, str]:
    """Initialize the image processor and model.
    
    Args:
        model_name (str): Name or path of the pretrained model
        
    Returns:
        Tuple[AutoProcessor, VitPoseForPoseEstimation, str]: A tuple containing:
            - image_processor: The initialized image processor
            - model: The initialized pose estimation model
            - device: The device identifier ('cuda', 'mps', or 'cpu')
    """
    device = get_device()
    image_processor = AutoProcessor.from_pretrained(model_name)
    model = VitPoseForPoseEstimation.from_pretrained(model_name, device_map=device)
    return image_processor, model, device

def normalize_bbox_coordinates(bbox: List[float], width: int, height: int) -> List[List[float]]:
    """Convert normalized bounding box coordinates to absolute coordinates.
    
    Args:
        bbox (List[float]): Normalized bounding box coordinates [x1, y1, x2, y2]
        width (int): Image width in pixels
        height (int): Image height in pixels
        
    Returns:
        List[List[float]]: Absolute bounding box coordinates in format [[x1, y1, x2, y2]]
    """
    return [[
        bbox[0] * width,
        bbox[1] * height,
        bbox[2] * width,
        bbox[3] * height
    ]]

def get_pose_estimation(
        image: Image.Image,
        bbox: List[List[float]],
        image_processor: AutoProcessor,
        model_name: str,
        model: VitPoseForPoseEstimation,
        confidence_threshold: float
        ) -> List:
    """Get pose estimation results for a single detection.
    
    Args:
        image (Image.Image): Input image
        bbox (List[List[float]]): Bounding box coordinates in format [[x1, y1, x2, y2]]
        image_processor (AutoProcessor): The initialized image processor
        model_name(str): Name of the model which is being used
        model (VitPoseForPoseEstimation): The pose estimation model
        confidence_threshold (float): Minimum confidence threshold for keypoint detection
        
    Returns:
        List: Processed pose estimation results containing keypoints and their scores
    """
    device = model.device.type
    inputs = image_processor(image, boxes=[bbox], return_tensors="pt").to(device)
    
    inputs["dataset_index"] = torch.tensor([0], device=device)

    with torch.no_grad():
        results = model(**inputs)
    
    return image_processor.post_process_pose_estimation(
        results, 
        boxes=[bbox],
        threshold=confidence_threshold
    )

def process_keypoints(
        pose_data: List,
        width: int,
        height: int,
        model_config: Dict
        ) -> List[fo.Keypoint]:
    """Process keypoints for a single person.
    
    Args:
        pose_data (List): Raw pose estimation results from the model
        width (int): Image width in pixels
        height (int): Image height in pixels
        model_config (Dict): Model configuration containing label mappings
        
    Returns:
        fo.Keypoint: FiftyOne Keypoint object containing all joints for a person
    """
    pose = pose_data[0][0]
    
    # Get all points and normalize to [0,1] range
    normalized_points = [(x/width, y/height) for x, y in pose['keypoints'].tolist()]
    scores = pose['scores'].tolist()
    
    # Create a single keypoint object for the person with all joint points
    person_keypoint = fo.Keypoint(
        label="person",
        points=normalized_points,
        confidence=scores,  # Store confidence for each joint
    )
    
    return person_keypoint

def process_sample(
        sample: fo.Sample,
        image_processor: AutoProcessor,
        model_name: str, 
        model: VitPoseForPoseEstimation,
        bbox_field: str,
        output_field: str,
        confidence_threshold: float
        ) -> None:
    """Process a single sample from the dataset.
    
    Args:
        sample (fo.Sample): The FiftyOne sample to process
        image_processor (AutoProcessor): The initialized image processor
        model_name(str): Name of the model which is being used
        model (VitPoseForPoseEstimation): The pose estimation model
        bbox_field (str): Field name containing bounding box detections
        output_field (str): Field name to save the keypoints to
        confidence_threshold (float): Confidence threshold for keypoint detection
        
    Returns:
        None: Results are saved directly to the sample
    """
    image = Image.open(sample.filepath)
    width = sample.get_field("metadata.width")
    height = sample.get_field("metadata.height")
    detected_people = sample.get_field(f"{bbox_field}.detections")
    
    model_results = []
    person_keypoints = []
    
    # Process each detected person
    for person in detected_people:
        abs_bbox = normalize_bbox_coordinates(person.bounding_box, width, height)

        pose_results = get_pose_estimation(
            image=image, 
            bbox=abs_bbox, 
            image_processor=image_processor, 
            model_name=model_name, 
            model=model, 
            confidence_threshold=confidence_threshold
            )
        
        model_results.append(pose_results)
    
    # Process results for all people
    for pose_data in model_results:
        keypoint = process_keypoints(pose_data, width, height, model.config)
        person_keypoints.append(keypoint)
    
    # Create the keypoints collection with the skeleton
    keypoints_collection = fo.Keypoints(keypoints=person_keypoints)
    
    # Save results
    sample[output_field] = keypoints_collection
    sample.save()

def run_pose_estimation(
        dataset: fo.Dataset,
        model_name: str,
        bbox_field: str,
        output_field: str,
        confidence_threshold: float
        ) -> None:
    """Process all samples in the dataset.
    Args:
        dataset (fo.Dataset): The FiftyOne dataset to process
        model_name (str): Name or path of the pretrained model
        bbox_field (str): Field name containing bounding box detections
        output_field (str): Field name to save the keypoints to
        confidence_threshold (float, optional): Confidence threshold for keypoint detection.
    Returns:
        None: Results are saved directly to the dataset samples
    """
    image_processor, model, _ = init_model(model_name)
    
    for sample in dataset.iter_samples():
        process_sample(sample, 
                       image_processor=image_processor, 
                       model=model, 
                       model_name=model_name,
                       bbox_field=bbox_field,
                       output_field=output_field,
                       confidence_threshold=confidence_threshold
                       )