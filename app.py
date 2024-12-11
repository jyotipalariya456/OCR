import streamlit as st
import easyocr
import cv2
import numpy as np
from deep_translator import GoogleTranslator
from PIL import Image
import io
import re
import torch
from diffusers import StableDiffusionPipeline
import speech_recognition as sr
import openai

# Initialize components
reader = easyocr.Reader(['en', 'hi'], gpu=False)
openai.api_key = "your_openai_api_key"  # Replace with your actual OpenAI API key
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")  # Use GPU

# Streamlit layout
st.title("Image Text Processing and Speech-to-Image Generation")

# File uploader for image processing
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
if uploaded_file:
    # Load and display image
    image = Image.open(uploaded_file)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # OCR Processing
    result = reader.readtext(img_cv)
    for detection in result:
        top_left = tuple(map(int, detection[0][0]))
        bottom_right = tuple(map(int, detection[0][2]))
        text = detection[1]
        img_cv = cv2.rectangle(img_cv, top_left, bottom_right, (0, 255, 0), 5)
        img_cv = cv2.putText(img_cv, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Extract text
    extracted_text = ' '.join([detection[1] for detection in result])
    st.subheader("Extracted Text:")
    st.write(extracted_text)

    # Language Detection and Translation
    def detect_language(text):
        return 'hi' if re.search(r'[\u0900-\u097F]', text) else 'en'

    translated_text = ""
    for sentence in re.split(r'(?<=[.!?]) +', extracted_text):
        lang = detect_language(sentence)
        if lang == 'hi':
            translated_text += GoogleTranslator(source='hi', target='en').translate(sentence) + " "
        else:
            translated_text += GoogleTranslator(source='en', target='hi').translate(sentence) + " "
    st.subheader("Translated Text:")
    st.write(translated_text)

    # Display processed image
    _, buffer = cv2.imencode('.png', img_cv)
    processed_image = Image.open(io.BytesIO(buffer))
    st.image(processed_image, caption='Processed Image with OCR', use_column_width=True)

# Speech-to-Image Generation
st.subheader("Speech-to-Image Generation")
if st.button("Generate Image from Speech"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening... Please speak your prompt.")
        try:
            audio = recognizer.listen(source, timeout=10)
            speech_text = recognizer.recognize_google(audio)
            st.write(f"Recognized Speech: {speech_text}")

            # Generate image from recognized speech
            with torch.no_grad():
                generated_image = pipe(speech_text).images[0]

            # Display and save the generated image
            st.image(generated_image, caption="Generated Image", use_column_width=True)
            generated_image.save("generated_image.png")
            st.success("Image saved as 'generated_image.png'.")
        except sr.UnknownValueError:
            st.error("Could not understand the audio. Please try again.")
        except sr.RequestError as e:
            st.error(f"Speech recognition service error: {e}")
        except Exception as e:
            st.error(f"Error during image generation: {e}")
