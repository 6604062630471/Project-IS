import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="IS Project", layout="wide")

st.markdown("""
    <style>
    /* Main button styling */
    div.stButton > button:first-child {
        background-color: #cc0000;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease 0s;
    }
    div.stButton > button:first-child:hover {
        background-color: #ff4b4b;
        box-shadow: 0px 5px 15px rgba(255, 75, 75, 0.4);
        transform: translateY(-2px);
    }
    /* Form container styling */
    div[data-testid="stForm"] {
        border-radius: 10px;
        border: 1px solid #333333;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Models ---
@st.cache_resource
def load_ml_models():
    model = joblib.load('water_ensemble_model.pkl')
    scaler = joblib.load('water_scaler.pkl')
    return model, scaler

@st.cache_resource
def load_nn_model():
    model = tf.keras.models.load_model('football_league_model.h5')
    return model

# --- Web Header ---
st.markdown("<h1 style='text-align: center; color: #ff4b4b; font-size: 3rem;'>⚡ Machine Learning & Neural Network</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888888; font-size: 1.2rem; margin-bottom: 2rem;'>Intelligent System Project</p>", unsafe_allow_html=True)

# --- Navigation Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "💧 1. ML Theory (Water)", 
    "🧪 2. Test ML", 
    "⚽ 3. NN Theory (Football Club Logo)", 
    "🔍 4. Test NN"
])

# ==========================================
# Tab 1: ML Theory (Water Potability)
# ==========================================
with tab1:
    st.subheader("💧 Water Potability Prediction System")
    
    with st.expander("1. Dataset Origin & Data Preprocessing", expanded=True):
        st.markdown("""
        * **Dataset Origin:** Sourced from Kaggle, containing 9 water quality metrics.
        * **Data Imperfection (Missing Values):** The dataset contained missing values (NaN) in the `ph`, `Sulfate`, and `Trihalomethanes` columns.
        * **Preprocessing Techniques:** * Applied **Data Imputation** by replacing missing values with the **Median** of their respective columns to prevent outlier distortion.
            * Applied **Feature Scaling** using `StandardScaler` to normalize the range of independent variables.
        """)
        
    with st.expander("2. Model Architecture (Ensemble Learning)"):
        st.markdown("""
        Implemented a **Voting Classifier** (Ensemble Learning) that aggregates the predictions of 3 distinct Machine Learning models:
        1. **Random Forest:** Handles non-linear data efficiently and reduces overfitting.
        2. **Support Vector Machine (SVM):** Finds the optimal hyperplane to classify complex dimensional data.
        3. **K-Nearest Neighbors (KNN):** Classifies data points based on the proximity of their chemical features.
        """)

# ==========================================
# Tab 2: Test ML (Water Potability)
# ==========================================
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    col_input, col_result = st.columns([2, 1], gap="large") 
    
    with col_input:
        st.markdown("### 📝 Input Chemical Features")
        with st.form("water_form"):
            c1, c2 = st.columns(2)
            with c1:
                ph = st.slider("💧 pH Level", 0.0, 14.0, 7.0, 0.1)
                Hardness = st.slider("🪨 Hardness", 40.0, 350.0, 150.0, 1.0)
                Solids = st.slider("🧂 Solids (Total dissolved)", 300.0, 60000.0, 20000.0, 100.0)
                Chloramines = st.slider("🧪 Chloramines", 0.0, 15.0, 7.0, 0.1)
                Sulfate = st.slider("🔬 Sulfate", 100.0, 500.0, 300.0, 1.0)
            with c2:
                Conductivity = st.slider("⚡ Conductivity", 150.0, 800.0, 400.0, 1.0)
                Organic_carbon = st.slider("🌿 Organic Carbon", 2.0, 30.0, 10.0, 0.1)
                Trihalomethanes = st.slider("☣️ Trihalomethanes", 0.0, 130.0, 60.0, 1.0)
                Turbidity = st.slider("🌫️ Turbidity", 1.0, 7.0, 4.0, 0.1)
                st.markdown("<br>", unsafe_allow_html=True)
                submit_button = st.form_submit_button("Analyze Water Quality ➔", use_container_width=True)
            
    with col_result:
        st.markdown("### 📊 Prediction Result")
        if submit_button:
            with st.spinner("AI is processing..."):
                model, scaler = load_ml_models()
                input_data = pd.DataFrame([[ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]], 
                                          columns=['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'])
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)
                
            with st.container(border=True): 
                if prediction[0] == 1:
                    st.success("🟢 **Status: Safe (Potable)**")
                    st.balloons()
                else:
                    st.error("🔴 **Status: Dangerous (Not Potable)**")
        else:
            with st.container(border=True):
                st.info("👈 Please adjust the chemical parameters on the left and click 'Analyze' to see the result.")

# ==========================================
# Tab 3: NN Theory (Logo Scanner)
# ==========================================
with tab3:
    st.subheader("⚽ Football Club Logo Classification (Top 5 Leagues)")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("1. Dataset & Image Preprocessing", expanded=True):
            st.markdown("""
            * **Dataset Origin:** Top 5 Football Leagues Club Logos from Kaggle.
            * **Data Imperfection:** Varying image resolutions and severely limited sample size (only 20 images per class).
            * **Preprocessing Techniques:** 1. **Image Resizing & Normalization:** Standardized all images to 128x128 pixels and normalized pixel values to a [0, 1] range.
                2. **Data Augmentation:** Implemented random flipping, rotation, and zooming to artificially expand the training dataset and prevent overfitting.
            """)
    with col2:
         with st.expander("2. Convolutional Neural Network (CNN) Structure", expanded=True):
            st.markdown("""
            * **Feature Extraction:** Utilized multiple `Conv2D` and `MaxPooling2D` layers to extract spatial hierarchies of features (e.g., edges, colors, shapes) from the logos.
            * **Classification Layer:** Flattened the extracted features and passed them through a `Dense` layer with a `Softmax` activation function to output the probability distribution across the 5 target leagues.
            """)
    with st.expander("3. Model Limitations & Future Work", expanded=False):
            st.markdown("""
            **Current Limitations (Why accuracy might be low):**
            * **Insufficient Data:** The model was trained on a very small dataset (only 20 images per league). Deep learning models typically require thousands of images to achieve high accuracy.
            * **Overfitting:** Due to the small sample size, the model may memorize the training images rather than learning generalizable patterns, leading to incorrect predictions on new data.
            * **High Complexity:** Football logos contain intricate details, text, and overlapping color schemes that are difficult for a basic CNN structure to distinguish.
            
            **Future Improvements:**
            * Collect a larger and more diverse dataset (500+ images per class).
            * Apply more advanced image preprocessing techniques (e.g., edge detection or color histogram equalization).
            """)

# ==========================================
# Tab 4: Test NN (Logo Scanner)
# ==========================================
with tab4:
    st.markdown("<br>", unsafe_allow_html=True)
    col_upload, col_result = st.columns([1, 1.2], gap="large") 
    
    with col_upload:
        st.markdown("### 📤 Upload Logo Image")
        with st.container(border=True):
            uploaded_file = st.file_uploader("Drag and drop or browse files (PNG, JPG)", type=['png', 'jpg', 'jpeg'])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption='Ready to scan', width=250)
                
    with col_result:
        st.markdown("### 🎯 AI Analysis")
        if uploaded_file is not None:
            st.write("Click the button below to feed the image into the CNN model.")
            if st.button("Scan Logo with AI ➔", use_container_width=True):
                with st.spinner('🔍 Extracting image features...'):
                    nn_model = load_nn_model()
                    
                    img_resized = image.resize((128, 128))
                    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    predictions = nn_model.predict(img_array)
                    score = tf.nn.softmax(predictions[0])
                    class_names = ['Bundesliga (Germany)', 'Ligue 1 (France)', 'La Liga (Spain)', 'Premier League (England)', 'Serie A (Italy)']
                    
                    predicted_class = class_names[np.argmax(score)]
                    
                with st.container(border=True):
                    st.caption("AI prediction result:")
                    st.markdown(f"<h2 style='color: #ff4b4b; text-align: center;'>🏆 {predicted_class}</h2>", unsafe_allow_html=True)
        else:
            with st.container(border=True):
                st.warning("👈 Please upload a football club logo image on the left to start the analysis.")