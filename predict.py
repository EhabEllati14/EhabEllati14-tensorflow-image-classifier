import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
from PIL import Image

def process_image(image):
    # Resize and normalize the image
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0
    return image.numpy()

def predict(image_path, model, top_k=5):
    # Load and process the image
    try:
        im = Image.open(image_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None

    im = np.asarray(im)
    processed_image = process_image(im)
    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(processed_image)
    probs = tf.math.top_k(predictions, k=top_k).values.numpy()[0]
    classes = tf.math.top_k(predictions, k=top_k).indices.numpy()[0]

    return probs, classes

def load_class_names(json_path):
    # Load class names from the JSON file
    with open(json_path, 'r') as f:
        class_names = json.load(f)
    return class_names

def display_results(image_path, probs, classes, class_names):
    # Display the image and predictions
    im = Image.open(image_path)
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(im)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.barh(range(len(probs)), probs, align='center')
    plt.yticks(range(len(probs)), [class_names[str(cls)] for cls in classes])
    plt.xlabel('Probability')
    plt.title('Top Predictions')
    plt.gca().invert_yaxis()
    plt.show()

def main():
    # Command-line arguments
    parser = argparse.ArgumentParser(description="Predict flower name from an image using a trained model.")
    parser.add_argument('image_path', type=str, help="Path to the image")
    parser.add_argument('model', type=str, help="Path to the trained model")
    parser.add_argument('top_k', type=int, help="Return the top K most likely classes")
    parser.add_argument('json', type=str, help="Path to the label map JSON file")

    args = parser.parse_args()

    # Load the model
    model = tf.keras.models.load_model(args.model, custom_objects={'KerasLayer': hub.KerasLayer})

    # Make predictions
    probs, classes = predict(args.image_path, model, args.top_k)

    # Load class names
    class_names = load_class_names(args.json)
    flower_names = [class_names[str(cls)] for cls in classes]

    # Print results
    print("Predicted flower names and probabilities:")
    for prob, flower_name in zip(probs, flower_names):
        print(f"{flower_name}: {prob:.4f}")

    # Display results
    display_results(args.image_path, probs, classes, class_names)

if __name__ == "__main__":
    main()

