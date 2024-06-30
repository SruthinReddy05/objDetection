import torch
from PIL import Image
import requests
from io import BytesIO

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


# Define a function to perform object detection
def detect_objects(image_path):
    # Load image
    img = Image.open(image_path)

    # Perform object detection
    results = model(img)

    # Extract detected objects and their bounding boxes
    detected_objects = results.xyxy[0]  # Extract the first (and only) set of detected objects

    # Print the detected objects and their bounding boxes
    print(detected_objects)

    # Display the annotated image with bounding boxes
    results.show()


# Example usage
def main():
    image_url = ('https://ewscripps.brightspotcdn.'
                 'com/dims4/default/5da4e86/2147483647/strip/true/crop/'
                 '1675x879+0+28/resize/1200x630!/quality/90/?url=http%3A%2F%2Fewscripps-brightspot.s3'
                 '.amazonaws.com%2F88%2Fa5%2Fbc16fd7044a0bf4f4c754b5c132b%2Ftraffic2.png')  # Replace this with your image URL
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img.save('example.jpg')  # Save the image locally
    detect_objects('example.jpg')  # Perform object detection on the saved image


if __name__ == "__main__":
    main()
