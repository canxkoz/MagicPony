# Quick Start

### Extract DINO Features
```
cd extract_dino
python extract.py -c configs/config.yml
```

## Filtering Data with YOLO-WORLD
To filter the data with YOLO-WORLD, you can use the following command:
```
cd filter_low_quality_augmentations
python filtering.py
```
*Note*: Update the path in image_folders (line 54) to point to the images you want to filter, and modify the path in new_folder_name (line 58) to specify where you want to save the filtered images. Additionally, you can adjust the threshold on line 78 to your desired value and change the class name on line 67 if needed.