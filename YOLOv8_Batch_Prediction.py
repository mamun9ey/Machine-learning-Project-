from ultralytics import YOLO
import torch
import cv2
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = YOLO('F:/road surface detect.v1i.yolov8/Succesful Train/weights/best.pt')
model.to(device)

if __name__ == '__main__':
    directory = 'F:/road surface detect.v1i.yolov8/datasets/train/images'
    files = os.listdir(directory)
    count = 0
    for index, file in enumerate(files):
        image = cv2.imread(os.path.join(directory, file))
        results = model(image)
        detections = results[0].boxes
        class_id = None
        for detection in detections:
            class_id = int(detection.cls[0])

        # cv2.imshow('image', results[0].plot())
        if class_id is None:
            cv2.imwrite(f'F:/road surface detect.v1i.yolov8/datasets/train/images/Detection/No_Detection{index}.jpg', results[0].plot())
            count += 1
        else:
            cv2.imwrite(f'F:/road surface detect.v1i.yolov8/datasets/train/images/Detection/{model.names[class_id]}_{index}.jpg', results[0].plot())

        print(f'Done {index + 1}/{len(files)}')
        # cv2.waitKey(0)
    print(f'No Detection Count: {count}')

    # cap = cv2.VideoCapture(0)
    # if not cap.isOpened():
    #     print('Cannot open camera')
    #     exit
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         print('Cannot get frame')
    #         break
    #
    #     results = model(frame)
    #     cv2.imshow('frame', results[0].plot())
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    cv2.destroyAllWindows()
