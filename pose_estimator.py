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

def initialize_pose_model(
    model_name: str,
    device: str = None
) -> Tuple[VitPoseForPoseEstimation, AutoProcessor, torch.device]:
    """Initialize pose estimation model and processor.
    
    Args:
        model_name: HuggingFace model identifier
        device: Optional device override
        
    Returns:
        Tuple of (model, processor, device)
    """
    if device is None:
        device = get_device()
    device = torch.device(device)
    
    processor = AutoProcessor.from_pretrained(model_name)
    model = VitPoseForPoseEstimation.from_pretrained(model_name).to(device)
    model.eval()
    return model, processor, device

def convert_to_absolute_bbox(
    bbox: List[float], 
    width: int, 
    height: int
) -> List[List[float]]:
    """Convert normalized bounding box coordinates to absolute pixel values.
    
    Args:
        bbox: Normalized coordinates [x1, y1, x2, y2] between 0 and 1
        width: Image width in pixels
        height: Image height in pixels
    
    Returns:
        List containing absolute coordinate bbox in format [[x1, y1, x2, y2]]
    """
    return [[
        bbox[0] * width,    # x1 to absolute
        bbox[1] * height,   # y1 to absolute
        bbox[2] * width,    # x2 to absolute
        bbox[3] * height    # y2 to absolute
    ]]

def get_pose_prediction(
    image: Image.Image,
    absolute_bbox: List[List[float]],
    model: AutoProcessor,
    image_processor: VitPoseForPoseEstimation,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Run pose estimation model inference on a single detection.
    
    Args:
        image: PIL Image object
        absolute_bbox: Bounding box in absolute coordinates [[x1, y1, x2, y2]]
        model: Pose estimation model
        image_processor: Image processor for model input preparation
        device: Torch device (CPU/GPU)
    
    Returns:
        Raw model output dictionary containing pose estimation results
    """
    # Prepare image and bbox for model input
    model_inputs = image_processor(
        image, 
        boxes=[absolute_bbox], 
        return_tensors="pt"
    ).to(device)
    
    # Run inference without gradient calculation
    with torch.no_grad():
        model_outputs = model(**model_inputs)
    
    return model_outputs

def process_single_detection(
    image: Image.Image,
    person_detection: fo.Detection,
    image_width: int,
    image_height: int,
    model: VitPoseForPoseEstimation,
    image_processor: AutoProcessor,
    device: torch.device,
    confidence_threshold: float
) -> List[Dict[str, torch.Tensor]]:
    """Process a single person detection through the pose estimation pipeline.
    
    Args:
        image: PIL Image object
        person_detection: FiftyOne Detection object containing bbox
        image_width: Image width in pixels
        image_height: Image height in pixels
        model: Pose estimation model
        image_processor: Image processor for model input preparation
        device: Torch device (CPU/GPU)
        confidence_threshold: Minimum confidence score for keypoint detection
    
    Returns:
        Processed pose estimation results containing keypoint coordinates and scores
    """
    # Convert detection bbox to absolute coordinates
    absolute_bbox = convert_to_absolute_bbox(
        person_detection.bounding_box, 
        image_width, 
        image_height
    )
    
    # Get model predictions
    model_outputs = get_pose_prediction(
        image, 
        absolute_bbox, 
        model, 
        image_processor, 
        device
    )
    
    # Post-process model outputs to get keypoint data
    processed_pose = image_processor.post_process_pose_estimation(
        model_outputs, 
        boxes=[absolute_bbox],
        threshold=confidence_threshold
    )
    
    return processed_pose

def create_keypoint_object(
    point_coordinates: Tuple[float, float], 
    confidence_score: float, 
    keypoint_label: str
) -> fo.Keypoint:
    """Create a single FiftyOne Keypoint object with normalized coordinates.
    
    Args:
        point_coordinates: Tuple of (x, y) normalized coordinates
        confidence_score: Detection confidence for this keypoint
        keypoint_label: String label identifying the keypoint type
    
    Returns:
        FiftyOne Keypoint object
    """
    return fo.Keypoint(
        label=keypoint_label,
        confidence=[confidence_score],
        points=[point_coordinates],
        skeleton=vitpose_skeleton
    )

def create_keypoints_from_pose(
    pose_result: List[Dict[str, torch.Tensor]],
    image_width: int,
    image_height: int,
    model_label_map: Dict[int, str]
) -> List[fo.Keypoint]:
    """Convert pose estimation results to FiftyOne keypoint objects.
    
    Args:
        pose_result: Processed pose estimation results
        image_width: Image width for coordinate normalization
        image_height: Image height for coordinate normalization
        model_label_map: Dictionary mapping keypoint indices to label names
    
    Returns:
        List of FiftyOne Keypoint objects
    """
    keypoints_list = []
    pose_data = pose_result[0][0]  # Extract single pose data
    
    # Convert tensor data to Python lists and normalize coordinates
    normalized_coordinates = [
        ((x/image_width), (y/image_height)) 
        for x, y in pose_data['keypoints'].tolist()
    ]
    confidence_scores = pose_data['scores'].tolist()
    keypoint_labels = [
        model_label_map[label_idx] 
        for label_idx in pose_data['labels'].tolist()
    ]
    
    # Create individual keypoint objects for each detected point
    for coordinates, score, label in zip(
        normalized_coordinates, 
        confidence_scores, 
        keypoint_labels
    ):
        keypoint = create_keypoint_object(coordinates, score, label)
        keypoints_list.append(keypoint)
    
    return keypoints_list

def run_pose_estimation(
    dataset: fo.Dataset,
    model_name: str,
    pose_field: str,
    bbox_detections:str, 
    device: str = None,
    confidence_threshold: float = 0.35
) -> None:
    """Process entire dataset to add pose keypoints to each sample.
    
    Args:
        dataset: FiftyOne dataset containing images and person detections
        model_name: HuggingFace model identifier
        pose_field: The field you want to store detections in
        bbox_detections: the field containing the bounding boxe
        image_processor: Image processor for model input preparation
        device: Optional device override
        confidence_threshold: Minimum confidence score for keypoint detection
    """
    model, processor, device = initialize_pose_model(model_name, device)

    for sample in dataset.iter_samples():
        # Load sample image and metadata
        image = Image.open(sample.filepath)
        image_width = sample.get_field("metadata.width")
        image_height = sample.get_field("metadata.height")
        person_detections = sample.get_field(f"{bbox_detections}.detections")
        
        # Process all person detections in the sample
        all_keypoints = []
        for person_detection in person_detections:
            # Get pose estimation results for single person
            pose_result = process_single_detection(
                image=image,
                person_detection=person_detection,
                image_width=image_width,
                image_height=image_height,
                model=model,
                image_processor=processor,
                device=device,
                confidence_threshold=confidence_threshold
            )
            
            # Convert pose results to keypoint objects
            person_keypoints = create_keypoints_from_pose(
                pose_result=pose_result,
                image_width=image_width,
                image_height=image_height,
                model_label_map=model.config.id2label
            )
            all_keypoints.extend(person_keypoints)
        
        # Update sample with all detected keypoints
        sample[pose_field] = fo.Keypoints(keypoints=all_keypoints)
        sample.save()
    
    # Ensure dataset is updated
    dataset.reload()