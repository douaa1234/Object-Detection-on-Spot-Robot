######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 11/11/22
# Description: 
# This program uses a TensorFlow Lite object detection model to perform object 
# detection on an image or a folder full of images. It draws boxes and scores 
# around the objects of interest in each image.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
"""import os
import cv2
import numpy as np
import sys
import glob
import importlib.util

# Paths and settings declared directly in the script
MODEL_DIR = '/Users/douaa/Desktop/spot/tflite1/custom_model_lite'  # Path to the model directory
GRAPH_NAME = 'detect.tflite'
LABELMAP_NAME = 'labelmap.txt'
IM_DIR = '/Users/douaa/Desktop/spot/spot-sdk/python/examples/get_image'  # Path to the directory containing images
CHECK_INTERVAL = 5  # Time interval to check for new images (in seconds)
min_conf_threshold = 0.5  # Confidence threshold
use_TPU = False  # Set to True if using Coral Edge TPU
save_results = True  # Set to True if you want to save the results
show_results = True  # Set to True if you want to show the results

# Import TensorFlow libraries
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    GRAPH_NAME = 'edgetpu.tflite'

# Get path to current working directory
CWD_PATH = os.getcwd()

# Define path to images and grab all image filenames
PATH_TO_IMAGES = os.path.join(CWD_PATH, IM_DIR)
images = glob.glob(PATH_TO_IMAGES + '/*.jpg') + glob.glob(PATH_TO_IMAGES + '/*.png') + glob.glob(PATH_TO_IMAGES + '/*.bmp')

if save_results:
    RESULTS_DIR = '/Users/douaa/Desktop/spot/tflite1/results'
    RESULTS_PATH = os.path.join(CWD_PATH, RESULTS_DIR)
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_DIR, GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_DIR, LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Remove '???' label for COCO starter model
if labels[0] == '???':
    del(labels[0])

# Load the TensorFlow Lite model
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

# Check output layer name to determine model type (TF1 or TF2)
outname = output_details[0]['name']
if 'StatefulPartitionedCall' in outname:  # TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:  # TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Loop over every image and perform detection
for image_path in images:
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape 
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform detection
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]  # Class index
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]  # Confidence scores

    detections = []

    # Loop over detections and draw box if confidence is above threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))
            
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            object_name = labels[int(classes[i])]  # Get object name
            label = f'{object_name}: {int(scores[i]*100)}%'  # Format label
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
            cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])

    # Show image
    if show_results:
        cv2.imshow('Object detector', image)
        image_fn = os.path.basename(image_path)
        image_savepath = os.path.join(RESULTS_PATH, image_fn)
        cv2.imwrite(image_savepath, image)

        cv2.waitKey(5000)
        cv2.destroyAllWindows()
        os.remove('/Users/douaa/Desktop/spot/spot-sdk/python/examples/get_image/frontleft_fisheye_image.jpg')


        
    
        

# Clean up
cv2.destroyAllWindows()
"""
######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 11/11/22
# Description: 
# This program uses a TensorFlow Lite object detection model to perform object 
# detection on an image or a folder full of images. It draws boxes and scores 
# around the objects of interest in each image.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import cv2
import numpy as np
import sys
import glob
import importlib.util
import time

# Paths and settings declared directly in the script
MODEL_DIR = '/Users/douaa/Desktop/spot/tflite1/custom_model_lite'  # Path to the model directory
GRAPH_NAME = 'detect.tflite'
LABELMAP_NAME = 'labelmap.txt'
IM_DIR = '/Users/douaa/Desktop/spot/spot-sdk/python/examples/get_image'  # Path to the directory containing images
CHECK_INTERVAL = 5  # Time interval to check for new images (in seconds)
min_conf_threshold = 0.5  # Confidence threshold
use_TPU = False  # Set to True if using Coral Edge TPU
save_results = True  # Set to True if you want to save the results
show_results = True  # Set to True if you want to show the results

XYZ_FILE_PATH = '/Users/douaa/Desktop/spot/xyz_coordinates.txt'

# Ensure that the file exists (create it if it doesn't)
if not os.path.exists(XYZ_FILE_PATH):
    with open(XYZ_FILE_PATH, 'w') as f:
        f.write("x, y, z\n")  # Add a header for clarity


# Import TensorFlow libraries
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    GRAPH_NAME = 'edgetpu.tflite'

# Get path to current working directory
CWD_PATH = os.getcwd()

# Define path to images and grab all image filenames
PATH_TO_IMAGES = os.path.join(CWD_PATH, IM_DIR)

# Create results directory if it doesn't exist
if save_results:
    RESULTS_DIR = '/Users/douaa/Desktop/spot/tflite1/results'
    RESULTS_PATH = os.path.join(CWD_PATH, RESULTS_DIR)
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_DIR, GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_DIR, LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Remove '???' label for COCO starter model
if labels[0] == '???':
    del(labels[0])

# Load the TensorFlow Lite model
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

# Check output layer name to determine model type (TF1 or TF2)
outname = output_details[0]['name']
if 'StatefulPartitionedCall' in outname:  # TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:  # TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2
    
# Search for depth image by filename containing the substring "depth"
depth_image_files = glob.glob(os.path.join(PATH_TO_IMAGES, '*depth*.png'))

# Check if depth image exists
if depth_image_files:
    DEPTH_IMAGE_PATH = depth_image_files[0]  # Use the first depth image found
    print(f'Found depth image: {DEPTH_IMAGE_PATH}')
else:
    raise FileNotFoundError("No depth image found in the directory with 'depth' in the filename.")

# Load depth image (assuming it's stored in 16-bit single-channel format)
depth_image = cv2.imread(DEPTH_IMAGE_PATH, cv2.IMREAD_UNCHANGED)

while True:
    # Define path to images and grab all image filenames
    images = glob.glob(PATH_TO_IMAGES + '/*.jpg') + glob.glob(PATH_TO_IMAGES + '/*.png') + glob.glob(PATH_TO_IMAGES + '/*.bmp')

    if not images:
        # Wait for CHECK_INTERVAL seconds before checking again
        time.sleep(CHECK_INTERVAL)
        continue

    # Process each image
    for image_path in images:
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imH, imW, _ = image.shape 
        image_resized = cv2.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)

        # Normalize pixel values if using a floating model
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform detection
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get detection results
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]  # Bounding box coordinates
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]  # Class index
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]  # Confidence scores

        detections = []
        xyz_coordinates = []

        # Loop over detections and draw box if confidence is above threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))

                # Calculate the center of the bounding box
                center_x = (xmin + xmax) // 2
                center_y = (ymin + ymax) // 2

                # Get z value from depth image at the bounding box center
                z = depth_image[center_y, center_x]

                # Add (x, y, z) to the list
                xyz_coordinates.append([center_x, center_y, z])

                # Store (x, y, z) in text file
                with open(XYZ_FILE_PATH, 'a') as f:
                    f.write(f'{center_x}, {center_y}, {z}\n')

                
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                object_name = labels[int(classes[i])]  # Get object name
                label = f'{object_name}: {int(scores[i]*100)}%'  # Format label
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
                cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])

        # Show image
        if show_results:
            cv2.imshow('Object detector', image)
            image_fn = os.path.basename(image_path)
            image_savepath = os.path.join(RESULTS_PATH, image_fn)
            cv2.imwrite(image_savepath, image)

            # Wait for 5 seconds, then delete the image
            cv2.waitKey(3000)
            cv2.destroyAllWindows()

            # Delete the image
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f'{image_path} has been deleted.')
            else:
                print(f'The file {image_path} does not exist.')

    # Wait for CHECK_INTERVAL seconds before checking for new images
    time.sleep(CHECK_INTERVAL)
