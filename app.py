import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import altair as alt
from inference import predict  # your model wrapper
import base64

# Streamlit page config
st.set_page_config(page_title="AgroVision", page_icon="🌱")

# Function to set local background image
def set_bg_image(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-attachment: fixed;
            background-size: cover;
            color: green;
        }}
        h1, h2, h3, h4, h5, h6, p, label {{
            color: green !important;
        }}
        .stButton>button {{
            background-color: #66BB6A;
            color: green;
            border-radius: 8px;
            padding: 0.5em 1em;
            font-weight: bold;
            border: none;
        }}
        .stFileUploader {{
            background-color: rgba(0, 0, 0, 0.4);
            border-radius: 8px;
            padding: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set your local background image
set_bg_image("imagenew1.jpeg")

# Disease to Solution Mapping for 7 classes
disease_solutions = {
    "Healthy": "The leaf looks healthy 🌿. No action needed. Continue good crop management.",
    "Others": "The disease could not be identified. Consider consulting an agronomist for accurate diagnosis.",
    "Tomato_Early_Blight": "Remove infected leaves, apply recommended fungicides, and ensure proper spacing between plants.",
    "anthracnose": "Prune affected parts, apply copper-based fungicides, and avoid overhead watering.",
    "cercospora_leaf_spot": "Remove infected leaves, apply fungicide, and rotate crops to reduce disease spread.",
    "phoshorus_deficiency": "Apply phosphorus-rich fertilizers according to soil test recommendations.",
    "rice_brown_leaf_spot": "Use resistant rice varieties, apply fungicides if necessary, and ensure proper field sanitation."
}

# App Title
st.title("AgroVision AI")
st.markdown(
    "<h5 style='text-align: center; font-style: italic;'>Empowering Farmers with AI for Healthier Crops</h5>",
    unsafe_allow_html=True
)

st.subheader("Upload a leaf image to predict possible diseases in crops")

st.markdown(
    """
    <style>
    .block-container {
        margin-left: 50%;   /* shift page slightly to the right */
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Leaf Image', use_column_width=True)

    if st.button("Analyze Image"):
        with st.spinner("🔎 Analyzing..."):
            # Predict
            pred_class, confidence, probs = predict(image)  # probs is dict of all classes

            # Display top prediction
            if "Healthy" in pred_class:
                st.success(f"The leaf is **{pred_class}** 🌿")
            elif 'Others': 
                "Sorry! This is out of this project scope"
            else:
                st.error(f"The leaf is affected by: **{pred_class}** 🚨")
            st.info(f"Confidence: **{confidence*100:.2f}%**")

            # Display recommendation
            solution = disease_solutions.get(pred_class, "Apply phosphorus-rich fertilizers according to soil test recommendations.")
            st.markdown("### 💡 Recommended Action:")
            st.markdown(solution)

            # Display full class probabilities as a bar chart
            df = pd.DataFrame({
                "Class": list(probs.keys()),
                "Probability": list(probs.values())
            })

            agriculture_colors = [
                "#2E7D32", "#66BB6A", "#A5D6A7",
                "#8D6E63", "#FBC02D", "#FFD54F", "#6D4C41"
            ]

            chart = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X("Class:N", sort="-y", title="Plant Disease Class"),
                    y=alt.Y("Probability:Q", title="Probability"),
                    color=alt.Color("Class:N", scale=alt.Scale(range=agriculture_colors)),
                    tooltip=["Class", alt.Tooltip("Probability", format=".2f")]
                )
            )

            text = (
                chart.mark_text(
                    align="center",
                    baseline="bottom",
                    dy=-2,
                    color="black",
                    fontWeight="bold"
                )
                .encode(
                    text=alt.Text("Probability:Q", format=".2f")
                )
            )

            st.write("### 📊 Class Probabilities")
            st.altair_chart(chart + text, use_container_width=True)































# Footer
st.markdown("---")
st.markdown("© 2025 AgroVision Project - Empowering Farmers with AI")
