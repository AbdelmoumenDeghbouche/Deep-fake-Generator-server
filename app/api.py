from fastapi import FastAPI, Response
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os

app = FastAPI()

LATENT_DIM = 32
CHANNELS = 3

# Load the TensorFlow Lite model
model_path = os.path.join(os.path.dirname(__file__), "generator.tflite")
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
@app.get("/")
def home():
    return {"message": "Hello World"}
@app.post("/generate_images")
async def generate_images(num_images: int):
    # Generate random latent vectors
    random_vectors = np.random.normal(size=(num_images, LATENT_DIM)).astype(np.float32) * 0.45

    generated_images = []
    for latent_vector in random_vectors:
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], [latent_vector])

        # Run the inference
        interpreter.invoke()

        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        generated_images.append(output_data[0])

    # Convert the generated images to bytes
    image_bytes = b''
    for img in generated_images:
        img = (img.squeeze() * 255).astype(np.uint8)
        img = Image.fromarray(img, mode='RGB')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        image_bytes += buffer.getvalue()

    return Response(content=image_bytes, media_type="image/png")
