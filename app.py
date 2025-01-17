import streamlit as st
import pandas as pd
from PIL import Image
import io
import datetime
import numpy as np
import torch
import monai
from monai.transforms import (
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensity,
    Resize,
    ToTensor
)
from monai.networks.nets import DenseNet121
from monai.visualize import blend_images
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoImageProcessor
import PIL.Image
import faiss
import sqlite3

# Set page configuration
st.set_page_config(
    page_title="Medical Imaging Dashboard",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state for storing patient data
if 'patient_records' not in st.session_state:
    st.session_state.patient_records = []

def save_patient_record(name, subscriber_id, image_type, image):
    """Save patient record to session state"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = {
        "timestamp": timestamp,
        "name": name,
        "subscriber_id": subscriber_id,
        "image_type": image_type,
        "image": image
    }
    st.session_state.patient_records.append(record)

def analyze_image(image_data):
    """Analyze medical image using MONAI and return insights"""
    try:
        # Ensure we're working with the raw data before normalization for stats
        stats = {
            "min_intensity": float(np.min(image_data)),
            "max_intensity": float(np.max(image_data)),
            "mean_intensity": float(np.mean(image_data)),
            "std_intensity": float(np.std(image_data))
        }
        
        # Generate histogram
        hist, bins = np.histogram(image_data.flatten(), bins=50)
        fig, ax = plt.subplots()
        ax.hist(image_data.flatten(), bins=50)
        ax.set_title("Intensity Distribution")
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Frequency")
        
        return stats, fig
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        return None, None

def process_medical_image(uploaded_file, image_type):
    """Process medical image using MONAI transforms"""
    try:
        # Read image data
        image = Image.open(uploaded_file)
        # Convert to grayscale if image is RGB
        if image.mode != 'L':
            image = image.convert('L')
        
        # Convert to numpy array and normalize to [0, 1] range
        image_data = np.array(image, dtype=np.float32)
        image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())
        
        # Add channel dimension if it doesn't exist
        if len(image_data.shape) == 2:
            image_data = image_data[np.newaxis, ...]  # Add channel dimension (1, H, W)
        
        # Define transforms based on image type
        transforms = monai.transforms.Compose([
            ScaleIntensity(),
            Resize((224, 224)),
            ToTensor()
        ])
        
        # Apply transforms
        processed_image = transforms(image_data)
        
        return image_data[0] if len(image_data.shape) == 3 else image_data, processed_image
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None

class ImageVectorStore:
    def __init__(self, db_path="image_store.db", vector_dim=768):
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(vector_dim)
        
        # Initialize SQLite database for metadata
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
        
    def create_tables(self):
        """Create necessary database tables if they don't exist."""
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    subscriber_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subscriber_id TEXT,
                    image_type TEXT,
                    image_path TEXT,
                    embedding_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (subscriber_id) REFERENCES users(subscriber_id)
                )
            """)
    
    def add_user(self, subscriber_id):
        """Add a new user to the system."""
        try:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO users (subscriber_id) VALUES (?)",
                    (subscriber_id,)
                )
            return True
        except sqlite3.IntegrityError:
            return False
    
    def add_image(self, subscriber_id, image_type, image_path, embedding):
        """Add a new image and its embedding to the store."""
        # Add embedding to FAISS index
        embedding = np.array([embedding]).astype('float32')
        embedding_id = self.index.ntotal
        self.index.add(embedding)
        
        # Add metadata to SQLite
        with self.conn:
            self.conn.execute("""
                INSERT INTO images 
                (subscriber_id, image_type, image_path, embedding_id)
                VALUES (?, ?, ?, ?)
            """, (subscriber_id, image_type, image_path, embedding_id))
    
    def get_user_images(self, subscriber_id):
        """Retrieve all images for a given user."""
        cursor = self.conn.execute("""
            SELECT image_type, image_path, embedding_id, created_at
            FROM images
            WHERE subscriber_id = ?
            ORDER BY created_at DESC
        """, (subscriber_id,))
        return cursor.fetchall()
    
    def find_similar_images(self, query_embedding, k=5):
        """Find k most similar images to the query embedding."""
        query_embedding = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        return distances[0], indices[0]
    
    def close(self):
        """Close the database connection."""
        self.conn.close()

# Main UI
st.title("Medical Imaging Dashboard 🏥")

# Sidebar for patient information input
with st.sidebar:
    st.header("Patient Information")
    patient_name = st.text_input("Patient Name")
    subscriber_id = st.text_input("Subscriber ID")
    image_type = st.selectbox("Image Type", ["MRI", "CT", "X-ray"])
    
    uploaded_file = st.file_uploader(
        "Upload medical image",
        type=["dcm", "png", "jpg", "jpeg"],
        help="Supported formats: DICOM, PNG, JPG, JPEG"
    )
    
    if st.button("Save Record"):
        if patient_name and subscriber_id and uploaded_file:
            # Process and analyze image
            image_data, processed_image = process_medical_image(uploaded_file, image_type)
            if image_data is not None:
                stats, hist_fig = analyze_image(image_data)
                
                # Save record with the numpy array directly
                record = {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "name": patient_name,
                    "subscriber_id": subscriber_id,
                    "image_type": image_type,
                    "image_data": image_data,  # Store numpy array directly
                    "stats": stats
                }
                st.session_state.patient_records.append(record)
                st.success("Record saved successfully!")
        else:
            st.error("Please fill in all fields and upload an image")

# Main content area
st.header("Patient Records")

# Display records in a tabular format
if st.session_state.patient_records:
    for idx, record in enumerate(st.session_state.patient_records):
        st.subheader(f"Patient: {record['name']} (ID: {record['subscriber_id']})")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("**Image Details:**")
            st.write(f"Image Type: {record['image_type']}")
            st.write(f"Timestamp: {record['timestamp']}")
            
            if 'stats' in record:
                st.write("**Image Statistics:**")
                st.write(f"Min Intensity: {record['stats']['min_intensity']:.2f}")
                st.write(f"Max Intensity: {record['stats']['max_intensity']:.2f}")
                st.write(f"Mean Intensity: {record['stats']['mean_intensity']:.2f}")
                st.write(f"Std Intensity: {record['stats']['std_intensity']:.2f}")
        
        with col2:
            # Display the numpy array directly
            st.image(record['image_data'], caption=f"{record['image_type']} Image", use_column_width=True)
else:
    st.info("No records found. Add a new record using the sidebar.")

# Add some CSS to improve the UI
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
    }
    .stExpander {
        background-color: #f0f2f6;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True) 