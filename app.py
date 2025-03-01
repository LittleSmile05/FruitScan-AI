import streamlit as st
import torch
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import time

# Set page title and layout
st.set_page_config(page_title="Real-Time Fruit Recognition", layout="wide")
st.title("Real-Time Fruit Recognition App")

# Load pre-trained model (MobileNet V2)
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.eval()
    return model

model = load_model()

# ImageNet class labels - directly embedded in the code
# This is a simplified version for the demo - in a real app you might want the full list
categories = [
    'tench', 'goldfish', 'great_white_shark', 'tiger_shark', 'hammerhead', 'electric_ray',
    'stingray', 'cock', 'hen', 'ostrich', 'brambling', 'goldfinch', 'house_finch', 'junco',
    # ... many other classes
    'plastic_bag', 'photocopier', 'coffee_mug', 'measuring_cup', 'teapot', 'hookah',
    'banana', 'strawberry', 'orange', 'lemon', 'fig', 'pineapple', 'apple', 'pomegranate', 'pear', 'grapes',
    'watermelon', 'mango', 'nectarine', 'kiwi', 'peach', 'coconut'
]

# Define fruit categories with their approximate indices in the model's output
# These indices might need adjustment for the actual model
fruit_categories = {
    'banana': 954,
    'apple': 948,
    'orange': 950,
    'lemon': 951,
    'pineapple': 953,
    'strawberry': 949,
    'pear': 956,
    'grape': 952,
    'watermelon': 958,
    'mango': 957
}

# Reverse mapping for display
idx_to_fruit = {v: k for k, v in fruit_categories.items()}

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Sidebar options
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
use_webcam = st.sidebar.checkbox("Use Webcam", True)
upload_option = st.sidebar.checkbox("Upload Image", False)

# Create a function to predict fruit
def predict_fruit(image):
    img = Image.fromarray(image)
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    
    with torch.no_grad():
        output = model(batch_t)
    
    # Get top probabilities and indices
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_indices = torch.topk(probabilities, 5)
    
    results = []
    fruit_results = []
    
    # Check if any of the top predictions are fruits in our list
    for i, (idx, prob) in enumerate(zip(top5_indices, top5_prob)):
        idx = idx.item()
        class_name = categories[idx] if idx < len(categories) else f"Class {idx}"
        probability = prob.item()
        
        results.append((class_name, probability))
        
        # Check if this class index is in our fruit mapping
        for fruit_idx in fruit_categories.values():
            if idx == fruit_idx and probability >= confidence_threshold:
                fruit_name = idx_to_fruit[idx]
                fruit_results.append((fruit_name.capitalize(), probability))
    
    return results, fruit_results

# Setup for webcam
if use_webcam:
    st.write("### Real-time Webcam Fruit Recognition")
    # Placeholder for webcam feed
    video_placeholder = st.empty()
    # Placeholder for results
    result_placeholder = st.empty()
    
    cap = cv2.VideoCapture(0)
    
    # Start button
    start_button = st.button("Start Webcam")
    stop_button = st.button("Stop Webcam")
    
    if start_button:
        st.session_state['webcam_running'] = True
    
    if stop_button:
        st.session_state['webcam_running'] = False
        cap.release()
    
    # Initialize webcam_running in session state if it doesn't exist
    if 'webcam_running' not in st.session_state:
        st.session_state['webcam_running'] = False
    
    # Webcam loop
    while st.session_state.get('webcam_running', False):
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image from webcam.")
            break
        
        # Convert BGR to RGB (OpenCV uses BGR, but we need RGB for display and processing)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Make prediction
        results, fruit_results = predict_fruit(frame_rgb)
        
        # Display webcam feed
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        
        # Display results
        result_text = "### Detected Fruits:\n"
        if fruit_results:
            for fruit_name, prob in fruit_results:
                result_text += f"- {fruit_name}: {prob:.2f}\n"
        else:
            result_text += "No fruits detected with sufficient confidence."
            
        result_placeholder.markdown(result_text)
        
        # Short delay to reduce CPU usage
        time.sleep(0.1)

# Image upload option
if upload_option:
    st.write("### Upload an image of a fruit")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load and display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Convert to RGB (in case it's RGBA or other format)
        image = image.convert("RGB")
        
        # Convert PIL Image to numpy array for prediction
        img_array = np.array(image)
        
        # Make prediction
        results, fruit_results = predict_fruit(img_array)
        
        # Display results
        st.write("### Detected Fruits:")
        if fruit_results:
            for fruit_name, prob in fruit_results:
                st.write(f"- {fruit_name}: {prob:.2f}")
        else:
            st.write("No fruits detected with sufficient confidence.")
        
        # Show all top predictions
        st.write("### All Top Predictions:")
        for class_name, prob in results:
            st.write(f"- {class_name}: {prob:.2f}")

# Instructions
st.sidebar.markdown("""
## Instructions
1. Choose to use webcam or upload an image
2. For webcam, click 'Start Webcam' to begin recognition
3. Adjust the confidence threshold as needed
4. Point camera at a fruit or upload a fruit image
""")