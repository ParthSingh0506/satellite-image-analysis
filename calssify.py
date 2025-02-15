import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Pre-trained ResNet50 Model for Classification
class LandUseClassifier(nn.Module):
    def __init__(self, num_classes=3):  # Example: Urban, Water, Vegetation
        super(LandUseClassifier, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Initialize and move to device
num_classes = 3
classifier = LandUseClassifier(num_classes).to(device)
classifier.eval()

# Preprocessing function for classification
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

# Object Detection (Using Faster R-CNN)
def detect_objects(image_path):
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval()

    # Load image
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Get predictions
    with torch.no_grad():
        predictions = model(image_tensor)

    # Extract results
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()

    return boxes, scores, labels

# Semantic Segmentation (DeepLabV3) with Overlay
def segment_image(image_path):
    model = models.segmentation.deeplabv3_resnet50(pretrained=True).to(device)
    model.eval()

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Perform segmentation
    with torch.no_grad():
        output = model(image_tensor)['out']
    
    # Convert to binary mask
    segmentation_map = output.argmax(dim=1).squeeze().cpu().numpy()

    # Convert segmentation mask to color
    segmentation_colored = np.uint8((segmentation_map / segmentation_map.max()) * 255)

    # Resize to match original image
    original_image = cv2.imread(image_path)
    segmentation_resized = cv2.resize(segmentation_colored, (original_image.shape[1], original_image.shape[0]))

    # Apply color mapping
    colored_mask = cv2.applyColorMap(segmentation_resized, cv2.COLORMAP_JET)

    # Blend segmentation mask with the original image
    blended_image = cv2.addWeighted(original_image, 0.6, colored_mask, 0.4, 0)

    # Save segmented image
    segmentation_output_path = "segmentation_output.jpg"
    cv2.imwrite(segmentation_output_path, blended_image)
    print(f"Segmentation result saved to {segmentation_output_path}")

# Function to overlay results on image
def visualize_results(image_path, predicted_class, boxes, scores):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (512, 512))

    # Add classification result
    class_names = ["Urban", "Water", "Vegetation"]
    label_text = f"Classification: {class_names[predicted_class]}"
    cv2.putText(image, label_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw object detection boxes
    for i, box in enumerate(boxes):
        if scores[i] > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, f"Object {i+1}: {scores[i]:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Save the output
    output_path = "output.jpg"
    cv2.imwrite(output_path, image)
    print(f"Results saved to {output_path}")

# Test the pipeline
image_path = "sample_satellite.jpg"  # Replace with an actual image path

# Classification
image_tensor = preprocess_image(image_path)
classification_output = classifier(image_tensor)
predicted_class = torch.argmax(classification_output, dim=1).item()

# Object Detection
boxes, scores, labels = detect_objects(image_path)

# Visualize classification & object detection
visualize_results(image_path, predicted_class, boxes, scores)

# Segmentation
segment_image(image_path)
