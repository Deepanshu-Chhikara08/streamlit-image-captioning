import streamlit as st  # Importing Streamlit for building the web application
from PIL import Image  # Importing PIL to handle image processing
from transformers import pipeline  # Importing pipeline from Hugging Face transformers for model loading

# Function to generate a text response from an image
def get_model_response(image_file):
    """
    Generate a caption for an input image using a pretrained model.

    Parameters:
    image_file: File object of the uploaded image.

    Returns:
    Generated text response or None if an error occurs.
    """
    try:
        # Initialize image-to-text model pipeline
        pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
        
        # Open image using PIL
        image = Image.open(image_file)
        
        # Generate caption from the image
        response = pipe(image)
        
        # Return generated text if response is successful
        return response[0]['generated_text']
    except Exception as e:
        # Display error message in the app if model call fails
        st.error(f"Error in AI model call: {e}")
        return None

# Function to process the uploaded image file for further use
def input_image_setup(uploaded_file):
    """
    Prepare the uploaded image file for processing.

    Parameters:
    uploaded_file: Streamlit uploaded file object.

    Returns:
    File object if available, otherwise None.
    """
    if uploaded_file:  
        # If file is uploaded, return it directly for processing
        return uploaded_file
    else:
        # Return None if no file is uploaded
        return None

# Streamlit app title and subtitle
st.title("Image to Text Model")
st.subheader("A model that can tell what is in an image...")

# File uploader widget for uploading an image file
uploaded_file = st.file_uploader("Upload a health-related image (optional)", type=["jpg", "jpeg", "png"])

# Button to trigger the image analysis
if st.button("Tell Me"):
    if uploaded_file:
        try:
            # Process the uploaded image
            image_file = input_image_setup(uploaded_file)
            
            # Prompt the AI model for generating a caption
            health_advice = get_model_response(image_file)

            if health_advice:
                # Display the generated caption
                st.write(health_advice)
            else:
                # Error message if AI model fails to respond
                st.error("Failed to get a response from the AI model.")
        except Exception as e:
            # Error message if any issue occurs in processing
            st.error(f"An error occurred while processing your request: {e}")
    else:
        # Warning message if no image file is uploaded
        st.warning("Please provide an image for analysis.")

# Display uploaded image in the app with a caption
if uploaded_file:
    # Open the uploaded image for preview
    image = Image.open(uploaded_file)
    
    # Display the image with a caption and column width fit
    st.image(image, caption="Uploaded Image", use_column_width=True)
