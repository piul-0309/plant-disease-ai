import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Plant Disease AI",
    page_icon="🌿",
    layout="centered"
)

# -------------------- CUSTOM STYLING --------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }

    div[data-testid="stFileUploader"] {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #2E8B57;
    }

    section[data-testid="stSidebar"] {
        background-color: #111111;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- LOAD MODEL --------------------
model = load_model("plant_disease_model.h5",compile=False)

# -------------------- CLASS NAMES --------------------
class_names = [
    "Apple___healthy",
    "Apple___Scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Corn___healthy",
    "Corn___Common_rust"
]

# -------------------- TITLE --------------------
st.markdown(
    "<h1 style='text-align:center; color:#2E8B57;'>🌿 Smart Plant Disease Detection System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>AI-powered crop health advisory tool</p>",
    unsafe_allow_html=True
)

st.divider()

# -------------------- SIDEBAR --------------------
st.sidebar.title("🌾 About This Project")
st.sidebar.write("""
This AI system detects plant diseases 
and recommends treatment solutions.

Developed using CNN and Streamlit.
""")

# -------------------- FILE UPLOADER --------------------
uploaded_file = st.file_uploader(
    "📸 Upload a Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------- MAIN LOGIC --------------------
if uploaded_file is not None:

    # Display uploaded image with border
    img = Image.open(uploaded_file)

    st.markdown(
        """
        <div style="
            border:3px solid #2E8B57;
            padding:10px;
            border-radius:15px;
            text-align:center;
        ">
        """,
        unsafe_allow_html=True
    )

    st.image(img, caption="Uploaded Leaf Image", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Preprocessing
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    confidence = float(np.max(prediction)) * 100
    predicted_class = class_names[np.argmax(prediction)]

    # -------------------- SEVERITY LOGIC --------------------
    if confidence > 85:
        severity = "🔴 High Severity"
    elif confidence > 65:
        severity = "🟡 Moderate Severity"
    else:
        severity = "🟢 Low / Early Stage"

    # -------------------- TREATMENT DICTIONARY --------------------
    treatment_dict = {
        "Apple___healthy": "The plant is healthy. Maintain proper irrigation and regular monitoring.",
        "Apple___Scab": "Apply fungicide like Captan. Remove infected leaves and improve air circulation.",
        "Apple___Black_rot": "Prune infected branches and apply copper-based fungicide.",
        "Apple___Cedar_apple_rust": "Use myclobutanil fungicide and remove nearby juniper plants.",
        "Corn___healthy": "Crop is healthy. Maintain fertilization and irrigation schedule.",
        "Corn___Common_rust": "Apply appropriate fungicide and consider resistant corn varieties."
    }

    treatment = treatment_dict.get(
        predicted_class,
        "Consult agricultural expert for treatment guidance."
    )

    st.divider()

    # -------------------- RESULT CARD --------------------
    st.markdown(
        f"""
        <div style="
            background-color:#1E1E1E;
            padding:25px;
            border-radius:18px;
            border:3px solid #2E8B57;
            box-shadow:0px 0px 25px rgba(46,139,87,0.6);
            text-align:center;
        ">
            <h2 style="color:#2E8B57;">🌿 AI Diagnosis Result</h2>
            <h3 style="color:white;">{predicted_class.replace('_',' ')}</h3>
            <p style="color:lightgray;">Confidence: {confidence:.2f}%</p>
            <p style="color:orange; font-weight:bold;">Severity: {severity}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Confidence bar
    st.progress(int(confidence))

    # -------------------- TREATMENT CARD --------------------
    st.markdown(
        f"""
        <div style="
            background-color:#262730;
            padding:20px;
            border-radius:12px;
            margin-top:15px;
        ">
            <h4 style="color:#4CAF50;">💊 Recommended Treatment</h4>
            <p style="color:white;">{treatment}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
