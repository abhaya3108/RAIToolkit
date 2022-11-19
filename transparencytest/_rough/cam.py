from enum import unique
import cv2
import imutils
import argparse
import numpy as np

from tensorflow import keras
from gradCAM import GradCAM
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils

def cam_result(model_file_path=None, image_file_name=None, folder_path=None, unique_id=None):
    loaded_model = keras.models.load_model(model_file_path)

    # load the input image
    print(image_file_name)
    image = load_img(image_file_name, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    orig = cv2.imread(image_file_name)

    preds = loaded_model.predict(image)
    i = np.argmax(preds[0])
    decoded = imagenet_utils.decode_predictions(preds)
    (imagenetID, label, prob) = decoded[0][0]

    cam = GradCAM(loaded_model, i)
    heatmap = cam.compute_heatmap(image)

    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

    # draw the predicted label on the output image
    cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
    cv2.putText(output, label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # display the original image and resulting heatmap and output image
    # to our screen
    original = f'{folder_path}\\images\\{unique_id}_original.png'
    heat_map = f'{folder_path}\\images\\{unique_id}_heat_map.png'
    cam_result = f'{folder_path}\\images\\{unique_id}_cam_result.png'

    cv2.imwrite(original, orig)
    cv2.imwrite(heat_map, heatmap)
    output = imutils.resize(output, height=700)
    cv2.imwrite(cam_result, output)
    print("Heat Map generated successfully!!!")

if __name__ == "__main__":
    # Create the parser
    my_parser = argparse.ArgumentParser(description='')

    # Add the arguments
    my_parser.add_argument('--model_file_path',
                        type=str,
                        help='model file path')
    # Add the arguments
    my_parser.add_argument('--image_file_name',
                        type=str,
                        help='Image file path')
    # Add the arguments
    my_parser.add_argument('--folder_path',
                        type=str,
                        help='File Saving Path')
    # Add the arguments
    my_parser.add_argument('--unique_id',
                        type=str,
                        help='File Saving ID')

    # Execute the parse_args() method
    args = my_parser.parse_args()

    model_file_path = args.model_file_path
    image_file_name = args.image_file_name
    folder_path = args.folder_path
    unique_id = args.unique_id
    # CAM result generation
    cam_result(model_file_path, image_file_name, folder_path, unique_id)