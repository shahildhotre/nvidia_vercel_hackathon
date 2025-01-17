import streamlit as st
import pandas as pd
from PIL import Image
import io
import datetime

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

# Main UI
st.title("Medical Imaging Dashboard 🏥")

# Sidebar for patient information input
with st.sidebar:
    st.header("Patient Information")
    patient_name = st.text_input("Patient Name")
    subscriber_id = st.text_input("Subscriber ID")
    image_type = st.selectbox("Image Type", ["MRI", "CT", "X-ray"])
    
    uploaded_file = st.file_uploader(
        "Drag and drop image here",
        type=["png", "jpg", "jpeg"],
        help="Supported formats: PNG, JPG, JPEG"
    )
    
    if st.button("Save Record"):
        if patient_name and subscriber_id and uploaded_file:
            image = Image.open(uploaded_file)
            save_patient_record(patient_name, subscriber_id, image_type, image)
            st.success("Record saved successfully!")
        else:
            st.error("Please fill in all fields and upload an image")

# Main content area
st.header("Patient Records")

# Display records in a tabular format
if st.session_state.patient_records:
    for idx, record in enumerate(st.session_state.patient_records):
        with st.expander(f"Record: {record['name']} - {record['timestamp']}"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write("**Patient Details:**")
                st.write(f"Name: {record['name']}")
                st.write(f"Subscriber ID: {record['subscriber_id']}")
                st.write(f"Image Type: {record['image_type']}")
                st.write(f"Timestamp: {record['timestamp']}")
            
            with col2:
                st.image(record['image'], caption=f"{record['image_type']} Image", use_column_width=True)
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