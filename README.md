# Violence Detection System using YOLO and CNN

This project is a real-time violence detection system implemented using the YOLO (You Only Look Once) object detection model and a Convolutional Neural Network (CNN) for classification.

## Requirements
- OpenCV (`cv2`)              command line-->pip install opencv-python
- NumPy                       command line-->pip install numpy
- Matplotlib                  command line-->pip install matplotlib
- TensorFlow                  command line-->pip install tensorflow
- Ultralytics YOLO library    command line-->pip install ultralytics
- Trained YOLO weights (`yolov8n.pt`)
- Trained CNN model (`CNN_mobilenetv2_model.h5`)
- COCO dataset classes file (`coco.names`)

## Setup
1. Install the required Python packages mentioned in requirements.

2. Place all the provided files in VIOLENCE DETECTION directory. 

## Execution

1. For `CNN_mobilenetv2_model.h5`, run the code `Image_Classifier.ipynb` file and save it in the directory.(before running download dataset from this link https://www.kaggle.com/datasets/swapneelbiswas/real-life-violence  and provide this file path mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5 in code).

2. Now save the `CNN_mobilenetv2_model.h5` file in the VIOLENCE DETECTION directory.

3. Run `Final_Code.ipynb` file to start the violence detection system(before running provide CNN_mobilenetv2_model.h5 file path in code and also provide any test video path). 

4.The system will continuously analyze the frames in real-time.

5. Detected objects will be classified as either "Violent" or "Non-Violent" based on the CNN model's predictions.

6. Bounding boxes will be drawn around detected objects, and labels will be displayed indicating the classification.

7. If a violent object is detected, a warning message will be printed.

8. Press 'q' to quit the application.

## Note
- Adjust the model paths and configurations as necessary.
