import cv2
import numpy as np
from sklearn.metrics import confusion_matrix

# Load the test image and run the object detection model
image = cv2.imread('test_image.jpg')
model = cv2.dnn.readNetFromTensorflow('model.pb')
blob = cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False)
model.setInput(blob)
output = model.forward()

# Extract the true and predicted labels
detection_annotations = ...  # Load the detection annotations
detection_classes = ...  # Load the list of detection classes

y_true = []
y_pred = []

for annotation in detection_annotations:
    true_label = detection_classes[annotation['class_id']]
    y_true.append(true_label)

    max_score = 0
    predicted_label = None

    # Find the predicted label with the highest confidence score
    for i in range(output.shape[2]):
        class_id = int(output[0, 0, i, 1])
        score = float(output[0, 0, i, 2])
        if score > max_score:
            predicted_label = detection_classes[class_id]
            max_score = score

    y_pred.append(predicted_label)

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=detection_classes)

print(cm)
