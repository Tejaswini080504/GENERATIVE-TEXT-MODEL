import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Function to load and preprocess the image
def load_image(image_path):
    img = Image.open(image_path)
    img = img.resize((256, 256))  # Resize for faster processing
    img = np.array(img) / 255.0  # Normalize to [0, 1]
    img = img.astype(np.float32)  # Ensure the image is float32
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img
# Function to display images
def display_images(content_image, style_image, generated_image):
    plt.figure(figsize=(12, 12))

    plt.subplot(1, 3, 1)
    plt.title("Content Image")
    plt.imshow(content_image[0])  # Remove batch dimension for display
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Style Image")
    plt.imshow(style_image[0])  # Remove batch dimension for display
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Generated Image")
    plt.imshow(generated_image[0])  # Remove batch dimension for display
    plt.axis('off')

    plt.show()

# Main function for Fast Style Transfer
def fast_style_transfer(content_path, style_path):
    # Load images
    content_image = load_image(content_path)
    style_image = load_image(style_path)

    # Load the pre-trained model from TensorFlow Hub
    model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1")

    # Perform style transfer
    stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]

    # Convert the generated image to a displayable format
    stylized_image = np.array(stylized_image) * 255.0
    stylized_image = np.clip(stylized_image, 0, 255).astype(np.uint8)

    return content_image, style_image, stylized_image

# User input for content and style images
content_path = input("Enter the path to the content image: ")
style_path = input("Enter the path to the style image: ")

# Run Fast Style Transfer
content_image, style_image, generated_image = fast_style_transfer(content_path, style_path)

# Display the results
display_images(content_image, style_image, generated_image)

# Save the generated image
output_path = input("Enter the path to save the styled image (e.g., styled_image.jpg): ")
# Remove the batch dimension before saving
Image.fromarray(generated_image[0]).save(output_path)
print(f"Styled image saved at {output_path}")
