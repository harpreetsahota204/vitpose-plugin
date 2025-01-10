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

def normalize_coordinates(
        coords: List[float], 
        width: int, 
        height: int
        ) -> List[float]:
    """Convert relative coordinates to absolute pixels."""
    return [
        coords[0] * width,  
        coords[1] * height,
        coords[2] * width,  
        coords[3] * height
    ]
def get_pose_prediction(
        image: Image.Image,
        bbox: List[float],
        model: VitPoseForPoseEstimation,
        processor: AutoProcessor,
        device: torch.device
        ) -> Dict[str, torch.Tensor]:
    """Run pose estimation inference."""
    inputs = processor(image, boxes=[bbox], return_tensors="pt").to(device)
    with torch.no_grad():
        return model(**inputs)

def process_detection(
        image: Image.Image,
        detection: fo.Detection,
        dims: Tuple[int, int],
        model: VitPoseForPoseEstimation,
        processor: AutoProcessor,
        device: torch.device,
        conf_threshold: float
        ) -> List[Dict[str, torch.Tensor]]:
    """Process single detection through pose estimation."""
    width, height = dims
    bbox = [normalize_coordinates(detection.bounding_box, width, height)]
    outputs = get_pose_prediction(image, bbox, model, processor, device)
    return processor.post_process_pose_estimation(
        outputs, boxes=[bbox], threshold=conf_threshold
    )

def create_keypoint(
        coords: Tuple[float, float], 
        score: float, 
        label: str
        ) -> fo.Keypoint:
    """Create a FiftyOne Keypoint object."""
    return fo.Keypoint(
        label=label,
        confidence=[score],
        points=[coords]
    )

def process_pose_results(
        pose_data: Dict[str, torch.Tensor],
        dims: Tuple[int, int],
        label_map: Dict[int, str]
        ) -> List[fo.Keypoint]:
    """Convert pose results to FiftyOne keypoints."""
    width, height = dims
    keypoints = []
    data = pose_data[0][0]
    
    coords = [(x/width, y/height) for x, y in data['keypoints'].tolist()]
    scores = data['scores'].tolist()
    labels = [label_map[idx] for idx in data['labels'].tolist()]
    
    for coord, score, label in zip(coords, scores, labels):
        keypoints.append(create_keypoint(coord, score, label))
    
    return keypoints

def run_pose_estimation(
        dataset: fo.Dataset,
        pose_field: str,
        model_name: str,
        device: str = None,
        conf_threshold: float = 0.35
        ) -> None:
    """Process dataset to add pose keypoints.
    
    Args:
        dataset: FiftyOne dataset containing images and person detections
        pose_field: Field name to store pose keypoints
        model_name: HuggingFace model identifier
        device: Optional device override
        conf_threshold: Minimum confidence score for keypoint detection
    """
    model, processor, device = initialize_pose_model(model_name, device)
    
    for sample in dataset.iter_samples():
        image = Image.open(sample.filepath)
        dims = (sample.metadata.width, sample.metadata.height)
        
        all_keypoints = []
        for detection in sample.bbox_detections.detections:
            pose_result = process_detection(
                image, detection, dims, model, processor, 
                device, conf_threshold
            )
            person_keypoints = process_pose_results(
                pose_result, dims, model.config.id2label
            )
            all_keypoints.extend(person_keypoints)
        
        sample[pose_field] = fo.Keypoints(keypoints=all_keypoints)
        sample.save()
    
    dataset.reload()