import os
import glob

import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

#Get bounding boxes from mask.
def get_bounding_box(ground_truth_map):
  # get bounding box from mask
  y_indices, x_indices = np.where(ground_truth_map > 0)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min)
  x_max = min(W, x_max)
  y_min = max(0, y_min)
  y_max = min(H, y_max)
  w = x_max - x_min
  h = y_max - y_min
  image_witdh = W
  image_height = H

  bbox = [x_min, y_min, w, h, image_witdh, image_height]

  return bbox


def add_padding_and_resize(image, bbox, target_size, padding):
    x, y, w, h, image_width, image_height = bbox
    aspect_ratio = w / h

    # Calculate padding to maintain aspect ratio
    if aspect_ratio > 1:
        padding_h = int((w - h) / 2) + padding
        padding_w = padding
    else:
        padding_w = int((h - w) / 2) + padding
        padding_h = padding

    new_x_min = max(0, x - padding_w)
    new_y_min = max(0, y - padding_h)
    new_x_max = min(image_width, x + w + padding_w)
    new_y_max = min(image_height, y + h + padding_h)

    cropped_image = image.crop((new_x_min, new_y_min, new_x_max, new_y_max))
    resized_image = cropped_image.resize((target_size, target_size), Image.LANCZOS)
    return resized_image

# get the list of image folders
image_folders = glob.glob("/YOUR-HOME-PATH/Wonder3D/bird_augmented_dataset/train/*")
image_folders.sort()

# new folder to save the filtered dataset
new_folder_name = "/YOUR-HOME-PATH/MagicPony/data/bird_augmented_dataset_filtered_v1"
if not os.path.exists(new_folder_name):
  os.makedirs(new_folder_name)

if not os.path.exists(new_folder_name + "/train"):
  os.makedirs(new_folder_name + "/train")

# load the YOLO-World model
model = YOLO('yolov8l-worldv2.pt')  # or choose yolov8m/l-world.pt
classes = ["bird"] # the class that we are interested in, in this case it is horse
model.set_classes(classes)

removed_folder_count = 0
padding = 20  # Padding amount

for i in range(len(image_folders)):
  count = 0
  image_paths = glob.glob(image_folders[i] + "/*rgb.png")
  for image_path in image_paths:
    results = model.predict(image_path, verbose=False)
    if len(results[0].boxes.data) == 0 or results[0].boxes.conf.cpu().numpy()[0] < 0.7:
      break
    count += 1
  if count == 6:
    image_paths_ = glob.glob(image_folders[i] + "/*rgb.png")
    new_image_folder_name = image_folders[i].split("/")[-1]

    if not os.path.exists(f"{new_folder_name}/train/{new_image_folder_name}"):
      os.makedirs(f"{new_folder_name}/train/{new_image_folder_name}")

    for image_path in image_paths_:
      mask_path = image_path.replace("rgb.png", "mask.png")
      dino_path = image_path.replace("rgb.png", "feat16.png")
      box_path = image_path.replace("rgb.png", "box.txt")

      image = Image.open(image_path)
      mask = Image.open(mask_path)
      mask = mask.convert("L")
      dino = Image.open(dino_path)

      # get bounding box from mask
      bbox = get_bounding_box(np.array(mask))
      # Add padding and resize the image
      resized_image = add_padding_and_resize(image, bbox, 256, padding)
      resized_mask = add_padding_and_resize(mask, bbox, 256, padding)

      # Save the resized image
      image_name = image_path.split("/")[-1]
      resized_image.save(f"{new_folder_name}/train/{new_image_folder_name}/{image_name}")

      mask_name = mask_path.split("/")[-1]
      resized_mask.save(f"{new_folder_name}/train/{new_image_folder_name}/{mask_name}")

      # Save the dino - dino has 6 64x64 images horizontally concatenated
      dinos = []
      x_min_dino = bbox[0] // 4
      x_max_dino = (bbox[0] + bbox[2]) // 4
      y_min_dino = bbox[1] // 4
      y_max_dino = (bbox[1] + bbox[3]) // 4
      padding_dino = padding // 4
      for j in range(6):
          dino_ = np.array(dino)[:, j * 64:(j + 1) * 64]
          dino_ = add_padding_and_resize(Image.fromarray(dino_), [x_min_dino, y_min_dino, x_max_dino - x_min_dino, y_max_dino - y_min_dino, 64, 64], 64, padding_dino)
          dino_ = np.array(dino_)
          dino_ = cv2.resize(dino_, (64, 64))
          dinos.append(dino_)

      dino = np.concatenate(dinos, axis=1)
      dino = cv2.cvtColor(dino, cv2.COLOR_RGB2BGR)
      dino_name = dino_path.split("/")[-1]
      cv2.imwrite(f"{new_folder_name}/train/{new_image_folder_name}/{dino_name}", dino)

      new_bbox = get_bounding_box(np.array(resized_mask))

      # New bounding box path
      with open(f"{new_folder_name}/train/{new_image_folder_name}/{box_path.split('/')[-1]}", "w") as f:
          f.write(str(dino_name.split("_")[0]) + " ")
          f.write(" ".join([str(i) for i in new_bbox]))
          f.write(" " + str(-1))
  else:
    removed_folder_count += 1

  if i % 1000 == 0:
      print(i)
print(removed_folder_count)