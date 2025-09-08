# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import pandas as pd
# import altair as alt
# from inference import predict  # your model wrapper
# import base64

# # Streamlit page config
# st.set_page_config(page_title="AgroVision", page_icon="🌱")

# # Function to set local background image
# def set_bg_image(image_file):
#     with open(image_file, "rb") as f:
#         data = f.read()
#     encoded = base64.b64encode(data).decode()
#     st.markdown(
#         f"""
#         <style>
#         .stApp {{
#             background-image: url("data:image/jpeg;base64,{encoded}");
#             background-attachment: fixed;
#             background-size: cover;
#             color: green;
#         }}
#         h1, h2, h3, h4, h5, h6, p, label {{
#             color: green !important;
#         }}
#         stButton>button {{
#         background-color: #66BB6A; /* Light green */
#         font-size: 100px; /* Bigger font */
#         padding: 15px 35px; /* Bigger button */
#         border-radius: 14px; /* More rounded corners */
#         font-weight: bold;
#         border: none;
#         transition: 0.3s;
    
#     }}
#         #     background-color: #66BB6A;
#         #     color: green;
#         #     border-radius: 8px;
#         #     padding: 0.5em 1em;
#         #     font-weight: bold;
#         #     border: none;
        
#         .stFileUploader {{
#             background-color: rgba(0, 0, 0, 0.4);
#             border-radius: 8px;
#             padding: 10px;
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

# # Set your local background image
# set_bg_image("mainimage.jpeg")

# # Disease to Solution Mapping for 7 classes
# disease_solutions = {
#     "Healthy": "The leaf looks healthy 🌿. No action needed. Continue good crop management.",
#     "Others": "The disease could not be identified. Consider consulting an agronomist for accurate diagnosis.",
#     "Tomato_Early_Blight": "Remove infected leaves, apply recommended fungicides, and ensure proper spacing between plants.",
#     "anthracnose": "Prune affected parts, apply copper-based fungicides, and avoid overhead watering.",
#     "cercospora_leaf_spot": "Remove infected leaves, apply fungicide, and rotate crops to reduce disease spread.",
#     "phoshorus_deficiency": "Apply phosphorus-rich fertilizers according to soil test recommendations.",
#     "rice_brown_leaf_spot": "Use resistant rice varieties, apply fungicides if necessary, and ensure proper field sanitation."
# }

# # App Title  
# st.markdown(
#     "<h1 style='text-align: center; color: green;'>AgroVision AI</h1>",
#     unsafe_allow_html=True
# )

# st.markdown(
#     "<h5 style='text-align: center; font-style: italic;'>Empowering Farmers with AI for Healthier Crops</h5>",
#     unsafe_allow_html=True
# )

# st.subheader("Upload a leaf image to predict possible diseases in crops")

# st.markdown(
#     """
#     <style>
#     .block-container {
#         margin-left: 10%;   /* shift page slightly to the right */
#         margin-right: auto;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Image uploader
# uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# if uploaded_image is not None:
#     # Open the uploaded image
#     image = Image.open(uploaded_image)

#     # Resize the image to reduce height while keeping aspect ratio
#     max_height = 400  # adjust this number as needed
#     ratio = max_height / image.height
#     new_width = int(image.width * ratio)
#     image = image.resize((new_width, max_height))

#     # Show the image in Streamlit
#     st.image(image, caption='Uploaded Leaf Image', use_container_width=False)


#     if st.button("Analyze Image"):
#         with st.spinner("🔎 Analyzing..."):
#             # Predict
#             pred_class, confidence, probs = predict(image)  # probs is dict of all classes

            
#             # Display top prediction
#         if pred_class == "Healthy":
#             st.success(f"The leaf is **{pred_class}** 🌿")
#         elif pred_class == "Others":
#             st.warning("Sorry! This is not part of this project scope 🚫")
#         else:
#             st.error(f"The leaf is affected by: **{pred_class}** 🚨")


#             # Display recommendation
#             solution = disease_solutions.get(
#                 pred_class,
#                 "Apply phosphorus-rich fertilizers according to soil test recommendations."
#             )
#             st.markdown("### 💡 Recommended Action:")
#             st.markdown(solution)

#             # # Display full class probabilities as a bar chart
#             # df = pd.DataFrame({
#             #     "Class": list(probs.keys()),
#             #     "Probability": list(probs.values())
#             # })

#             # agriculture_colors = [
#             #     "#2E7D32", "#66BB6A", "#A5D6A7",
#             #     "#8D6E63", "#FBC02D", "#FFD54F", "#6D4C41"
#             # ]

#             # chart = (
#             #     alt.Chart(df)
#             #     .mark_bar()
#             #     .encode(
#             #         x=alt.X("Class:N", sort="-y", title="Plant Disease Class"),
#             #         y=alt.Y("Probability:Q", title="Probability"),
#             #         color=alt.Color("Class:N", scale=alt.Scale(range=agriculture_colors)),
#             #         tooltip=["Class", alt.Tooltip("Probability", format=".2f")]
#             #     )
#             # )

#             # text = (
#             #     chart.mark_text(
#             #         align="center",
#             #         baseline="bottom",
#             #         dy=-2,
#             #         color="black",
#             #         fontWeight="bold"
#             #     )
#             #     .encode(
#             #         text=alt.Text("Probability:Q", format=".2f")
#             #     )
#             # )

#             # st.write("### 📊 Class Probabilities")
#             # st.altair_chart(chart + text, use_container_width=True)

# # Footer
# st.markdown("---")
# st.markdown("© 2025 AgroVision Project - Empowering Farmers with AI")
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
        /* Background styling */
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-attachment: fixed;
            background-size: cover;
            color: green;
        }}

        /* Global text color */
        h1, h2, h3, h4, h5, h6, p, label {{
            color: green !important;
        }}

        /* Upload button text change */
        .stFileUploader label::after {{
            content: "Upload Image";  /* Change Browse File to Upload Image */
        }}

        /* Analyze Image Button Styling */
        div.stButton > button {{
            font-size: 30px !important;        
            background-color: white !important;  
            color: #1B5E20 !important;           
            padding: 20px 50px !important;       
            border-radius: 50px !important;      
            border: 2px solid #1B5E20 !important;

        
            font-weight: bold !important;
            transition: 0.3s;
        }}
        div.stButton > button:hover {{
            background-color: #f1f1f1 !important;
            color: #2E7D32 !important;             
        }}

        /* Top Prediction Text */
        .prediction-text {{
            font-size: 20px !important;
            font-weight: bold;
            color: #1B5E20;
        }}

        /* Recommendation Text */
        .recommendation-text {{
            font-size: 20px !important;
            line-height: 1.6;
            color: #1B5E20;
            font-weight: bold;
        }}

        /* Page container margins */
        .block-container {{
            margin-left: 5%;  
            margin-right: 10%;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set your local background image
set_bg_image("mainimage.jpeg")

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
st.markdown("<h1 style='text-align: center; color: green;'>AgroVision AI</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; font-style: italic;'>Empowering Farmers with AI for Healthier Crops</h5>", unsafe_allow_html=True)

st.subheader("Upload a leaf image to predict possible diseases in crops")
st.markdown(
     """
     <style>
     .block-container {
         margin-left: 10%;   /* shift page slightly to the right */
         margin-right: auto;
     }
     </style>
     """,
     unsafe_allow_html=True
 )

# Image uploader
uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"])  # Empty label since we styled it

if uploaded_image is not None:
    # Open and resize the uploaded image
    image = Image.open(uploaded_image)
    max_height = 400
    ratio = max_height / image.height
    new_width = int(image.width * ratio)
    image = image.resize((new_width, max_height))
    st.image(image, caption='Uploaded Leaf Image', use_container_width=False)

    if st.button("Analyze Image"):
        with st.spinner("🔎 Analyzing..."):
            pred_class, confidence, probs = predict(image)

            # Display top prediction with bigger text
            if pred_class == "Healthy":
                st.markdown(f"<p class='prediction-text'>The leaf is <b>{pred_class}</b> 🌿</p>", unsafe_allow_html=True)
            elif pred_class == "Others":
                st.markdown("<p class='prediction-text'>Sorry! This is not part of this project scope 🚫</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='prediction-text'>The leaf is affected by: <b>{pred_class}</b> 🚨</p>", unsafe_allow_html=True)

            # Display recommendation with bigger text
            solution = disease_solutions.get(pred_class, "Apply phosphorus-rich fertilizers according to soil test recommendations.")
            st.markdown("### 💡 Recommended Action:")
            st.markdown(f"<p class='recommendation-text'>{solution}</p>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("© 2025 AgroVision Project - Empowering Farmers with AI")

