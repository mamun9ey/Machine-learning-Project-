from ultralytics import YOLO

# Create a Custom YAML Configuration (adjust paths as needed)
data = """
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 9
names: ['High Edge Cracking', 'High Pothole', 'High Ravelling', 'Low Edge Cracking', 'Low Pothole', 'Low Ravelling', 'Medium Edge Cracking', 'Medium Pothole', 'Medium Ravelling']

roboflow:
  workspace: college-7qowe
  project: pavement-distress-datasets
  version: 4
  license: Public Domain
  url: https://universe.roboflow.com/college-7qowe/pavement-distress-datasets/dataset/4
"""

with open('data.yaml', 'w') as file:
    file.write(data)

model = YOLO('yolov8n.pt')

if __name__ == '__main__':
    # Train the model
    model.train(data='data.yaml', epochs=5, imgsz=2358)
    model.val(data='data.yaml')
    model.save('best_model.pt')
    print('Model saved')

