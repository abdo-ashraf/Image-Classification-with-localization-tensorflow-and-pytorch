## Update: added classification-with-localization-pytorch-with-augmentation.ipynb
- which contains pytorch model trained on whole caltech dataset with better performance and augmentation used using albumentation package
# Image-Classification-with-localization
- In this project i did Classification with localization (aka object localization)
- I used portion of Caltech-101 dataset this protion was ['airplanes', 'Faces', 'Motorbikes']
- Version 2: 
  - functions that I used to parse data and show image all in my_helper_funcs.py file
  - I trained two transfer learning models:
    - first one was with NasNet as base model
      - training class accuracy: 100%
      - training bounding box loss: 1551.15
      - Validation accuracy: 100%
      - training bounding box loss: 1293.05
    - second model was with VGG-16 as base model
      - training class accuracy: 0.9984%
      - training bounding box loss: 688.58
      - Validation accuracy: 100%
      - training bounding box loss: 802.32
- Version 3:
  - https://drive.google.com/drive/folders/1I55CnyZbGj2Gsg5IVqeIaKitRl8WpA97?usp=drive_link  
## - Note: I did not do image Augmentation yet
