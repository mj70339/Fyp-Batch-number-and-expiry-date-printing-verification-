import serial
import time
import torch
import cv2
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Load custom YOLOv5 model
path = r'C:\FYPN\yolov5'
sys.path.insert(0, path)

from utils.torch_utils import select_device
from utils.general import non_max_suppression
from models.common import DetectMultiBackend

device = select_device('')
model = DetectMultiBackend(r'C:\FYPN/bestfinal.pt', device=device)
model.eval()


cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

ser = serial.Serial('COM4', 115200)

# Thresholds
CONF_THRESHOLD = 0.7
PRINT_THRESHOLD = 5

# Counters
count_print = 0
count_misPrint = 0
count_no_detection = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    frame = cv2.resize(frame, (512, 512))  # width=512, height=512
    img_tensor = torch.from_numpy(frame).permute(
        2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor)
        pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.45)


    detected = False
    
    for det in pred:
        if det is not None and len(det):
            for *xyxy, conf, cls in det:
                class_id = int(cls.item())
                confidence = float(conf.item())

                if class_id == 0 and confidence > 0.5:
                    count_misPrint += 1
                    detected = True
                    break  # exit inner loop
                elif class_id == 1 and confidence > CONF_THRESHOLD:
                    count_print += 1
                    detected = True
                    break  # exit inner loop
        if detected:
            break


    if count_print >= PRINT_THRESHOLD:
        print('0')  # Correct print
        count_print = 0
        count_misPrint = 0
        count_no_detection = 0
        ser.write(b'0')
        time.sleep(5)

    elif count_misPrint >= PRINT_THRESHOLD:
        print('1')  # Misprint
        count_misPrint = 0
        count_print = 0
        count_no_detection = 0
        ser.write(b'1')

    elif not detected:
        count_no_detection += 1
        if count_no_detection >= PRINT_THRESHOLD:
            print('1')  # No detection after enough attempts
            count_no_detection = 0
            count_print = 0
            count_misPrint = 0
            time.sleep(2)
            ser.write(b'1')
    else:
        # Reset no-detection counter if something was detected
        count_no_detection = 0

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
ser.close()
cv2.destroyAllWindows()
