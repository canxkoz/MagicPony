import os
import glob

from ultralytics import YOLO

# get the list of image folders
image_folders = glob.glob("/YOUR-PATH/augmented_dataset/train/*")
image_folders.sort()

# load the YOLO-World model
model = YOLO('yolov8l-worldv2.pt')  # or choose yolov8m/l-world.pt
classes = ["horse"] # the class that we are interested in, in this case it is horse
model.set_classes(classes)

removed_folder_count = 0
for i in range(len(image_folders)):
  count = 0
  image_paths = glob.glob(image_folders[i] + "/*.jpg")
  print("Folder: ", image_folders[i])
  for image_path in image_paths:
    results = model.predict(image_path)
    print("Count of horses detected in the view: ", len(results[0].boxes.data))
    if len(results[0].boxes.data) == 0 or results[0].boxes.conf.cpu().numpy()[0] < 0.1:
      break
    count += 1
  if count == 6:
    # copy folder to filtered_dataset
    if not os.path.exists("/YOUR-PATH/filtered_dataset"):
      os.mkdir("/YOUR-PATH/filtered_dataset")
    os.system(f"cp -r {image_folders[i]} /YOUR-PATH/filtered_dataset")
  else:
    removed_folder_count += 1
print(removed_folder_count)