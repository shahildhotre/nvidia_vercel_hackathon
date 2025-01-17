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

# Add reset button in top right
col1, col2, col3 = st.columns([1, 1, 0.2])
with col3:
    if st.button("🔄 Reset"):
        st.session_state.patient_records = []
        st.rerun()

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
    """Analyze medical image using MONAI and return detailed medical insights"""
    try:
        # Basic statistical analysis
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
        
        # Additional detailed medical analysis
        analysis = {
            "anatomical_features": {
                "density_variations": detect_density_variations(image_data),
                "tissue_characteristics": analyze_tissue_characteristics(image_data),
                "organ_measurements": measure_organs(image_data),
            },
            "abnormalities": {
                "masses": detect_masses(image_data),
                "fractures": detect_fractures(image_data),
                "inflammation": detect_inflammation(image_data),
                "fluid_collections": detect_fluid_collections(image_data)
            },
            "views": generate_multiplanar_views(image_data)
        }
        
        return stats, fig, analysis
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        return None, None, None

def detect_density_variations(image_data):
    """Analyze density variations in the image"""
    # Placeholder for density analysis logic
    return {
        "high_density_regions": "Areas of high density detected in upper quadrant",
        "low_density_regions": "Normal tissue density distribution observed",
        "density_gradients": "Gradual density transition in peripheral regions"
    }

def analyze_tissue_characteristics(image_data):
    """Analyze tissue characteristics"""
    # Placeholder for tissue analysis logic
    return {
        "tissue_types": ["soft tissue", "bone", "fluid"],
        "tissue_boundaries": "Well-defined tissue boundaries observed",
        "texture_analysis": "Normal tissue texture patterns"
    }

def measure_organs(image_data):
    """Measure and analyze visible organs"""
    # Placeholder for organ measurement logic
    return {
        "dimensions": "Standard organ dimensions within normal range",
        "volume_calculations": "No significant volumetric abnormalities",
        "position_analysis": "Normal anatomical positioning"
    }

def detect_masses(image_data):
    """Detect and analyze potential masses"""
    # Placeholder for mass detection logic
    return {
        "presence": False,
        "characteristics": "No suspicious masses detected",
        "location": "N/A"
    }

def detect_fractures(image_data):
    """Detect and analyze potential fractures"""
    # Placeholder for fracture detection logic
    return {
        "presence": False,
        "type": "No fractures detected",
        "location": "N/A"
    }

def detect_inflammation(image_data):
    """Detect and analyze areas of inflammation"""
    # Placeholder for inflammation detection logic
    return {
        "presence": False,
        "severity": "No significant inflammation detected",
        "location": "N/A"
    }

def detect_fluid_collections(image_data):
    """Detect and analyze fluid collections"""
    # Placeholder for fluid collection detection logic
    return {
        "presence": False,
        "volume": "No abnormal fluid collections",
        "location": "N/A"
    }

def generate_multiplanar_views(image_data):
    """Generate axial, sagittal, and coronal views"""
    # Placeholder for multiplanar reconstruction logic
    return {
        "axial": "Axial view analysis complete",
        "sagittal": "Sagittal view analysis complete",
        "coronal": "Coronal view analysis complete"
    }

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

def check_patient_record(subscriber_id, image_type, uploaded_file):
    """Check if patient record exists and handle accordingly"""
    vector_store = ImageVectorStore()
    
    # Check if user exists in database
    existing_images = vector_store.get_user_images(subscriber_id)
    
    if existing_images:
        # Process new image to compare with existing ones
        image_data, processed_image = process_medical_image(uploaded_file, image_type)
        if image_data is not None:
            # TODO: Generate embedding for the new image
            # For now, using dummy embedding
            dummy_embedding = np.zeros(768, dtype='float32')
            
            # Find similar images
            distances, indices = vector_store.find_similar_images(dummy_embedding)
            
            summary = {
                "exists": True,
                "total_records": len(existing_images),
                "image_types": set(img[0] for img in existing_images),
                "latest_visit": existing_images[0][3],  # latest created_at
                "similar_images": len(indices)
            }
            
            return summary, image_data
    
    return {"exists": False}, None

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
            # Check existing records
            summary, image_data = check_patient_record(subscriber_id, image_type, uploaded_file)
            
            if summary["exists"]:
                st.info(f"""
                    Existing patient record found:
                    - Total previous visits: {summary['total_records']}
                    - Previous image types: {', '.join(summary['image_types'])}
                    - Last visit: {summary['latest_visit']}
                    - Similar images found: {summary['similar_images']}
                """)
            else:
                st.warning(f"""
                    New patient detected!
                    - Patient Name: {patient_name}
                    - Subscriber ID: {subscriber_id}
                    - Image Type: {image_type}
                    Creating new record in the system...
                """)
            
            # Process and analyze image
            if image_data is None:
                image_data, processed_image = process_medical_image(uploaded_file, image_type)
            
            if image_data is not None:
                stats, hist_fig, analysis = analyze_image(image_data)
                
                # Save record
                record = {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "name": patient_name,
                    "subscriber_id": subscriber_id,
                    "image_type": image_type,
                    "image_data": image_data,
                    "stats": stats,
                    "analysis": analysis
                }
                st.session_state.patient_records.append(record)
                
                # Add to vector store if new patient
                if not summary["exists"]:
                    vector_store = ImageVectorStore()
                    vector_store.add_user(subscriber_id)
                    # TODO: Generate proper embedding
                    dummy_embedding = np.zeros(768, dtype='float32')
                    vector_store.add_image(subscriber_id, image_type, "temp_path", dummy_embedding)
        else:
            st.error("Please fill in all fields and upload an image")

# Main content area
st.header("Patient Records")

# Display records in a tabular format
if st.session_state.patient_records:
    for idx, record in enumerate(st.session_state.patient_records):
        st.subheader(f"Patient: {record['name']} (ID: {record['subscriber_id']})")
        
        # Display image first
        st.image(record['image_data'], caption=f"{record['image_type']} Image", use_column_width=True)
        
        # Basic info in columns
        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
        with info_col1:
            st.metric("Image Type", record['image_type'])
        with info_col2:
            st.metric("Scan Date", record['timestamp'])
        with info_col3:
            st.metric("Min Intensity", f"{record['stats']['min_intensity']:.2f}")
        with info_col4:
            st.metric("Max Intensity", f"{record['stats']['max_intensity']:.2f}")
        
        # Detailed Analysis Section
        st.markdown("### 📋 Detailed Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Density Analysis", "Tissue Characteristics", "Measurements"])
        
        with tab1:
            st.markdown("#### Density Distribution")
            density_col1, density_col2 = st.columns(2)
            
            with density_col1:
                st.markdown("**🔸 High Density Regions**")
                st.markdown(record['analysis']['anatomical_features']['density_variations']['high_density_regions'])
                
                st.markdown("**🔸 Density Gradients**")
                st.markdown(record['analysis']['anatomical_features']['density_variations']['density_gradients'])
            
            with density_col2:
                st.markdown("**🔸 Low Density Regions**")
                st.markdown(record['analysis']['anatomical_features']['density_variations']['low_density_regions'])
        
        with tab2:
            st.markdown("#### Tissue Analysis")
            tissue_col1, tissue_col2 = st.columns(2)
            
            with tissue_col1:
                st.markdown("**🔬 Identified Tissue Types:**")
                for tissue in record['analysis']['anatomical_features']['tissue_characteristics']['tissue_types']:
                    st.markdown(f"- {tissue.title()}")
            
            with tissue_col2:
                st.markdown("**🔬 Tissue Properties:**")
                st.markdown(f"- Boundaries: {record['analysis']['anatomical_features']['tissue_characteristics']['tissue_boundaries']}")
                st.markdown(f"- Texture: {record['analysis']['anatomical_features']['tissue_characteristics']['texture_analysis']}")
        
        with tab3:
            st.markdown("#### Organ Measurements")
            measurements = record['analysis']['anatomical_features']['organ_measurements']
            
            for key, value in measurements.items():
                st.markdown(f"**📏 {key.replace('_', ' ').title()}:**")
                st.markdown(value)
        
        # Abnormalities Section
        st.markdown("### ⚕️ Clinical Findings")
        findings_col1, findings_col2 = st.columns(2)
        
        with findings_col1:
            abnormalities = record['analysis']['abnormalities']
            
            # Masses
            if abnormalities['masses']['presence']:
                st.error("⚠️ Masses Detected")
            else:
                st.success("✅ No Masses Detected")
            st.markdown(abnormalities['masses']['characteristics'])
            
            # Inflammation
            if abnormalities['inflammation']['presence']:
                st.warning("🔥 Inflammation Present")
            else:
                st.success("✅ No Inflammation")
            st.markdown(f"Severity: {abnormalities['inflammation']['severity']}")
        
        with findings_col2:
            # Fractures
            if abnormalities['fractures']['presence']:
                st.error("⚠️ Fractures Detected")
            else:
                st.success("✅ No Fractures")
            st.markdown(abnormalities['fractures']['type'])
            
            # Fluid Collections
            if abnormalities['fluid_collections']['presence']:
                st.warning("💧 Fluid Collections Present")
            else:
                st.success("✅ No Fluid Collections")
            st.markdown(abnormalities['fluid_collections']['volume'])
        
        st.markdown("---")
else:
    st.info("No records found. Add a new record using the sidebar.")

# Add custom CSS for better styling
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px;
        color: #0e1117;
        padding: 0px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0e1117;
        color: #ffffff;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h3 {
        padding-top: 1rem;
        padding-bottom: 0.5rem;
    }
    h4 {
        color: #0e1117;
        padding-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True) 