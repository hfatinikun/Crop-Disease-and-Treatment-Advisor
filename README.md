# Crop-Disease-and-Treatment-Advisor
Identification of plant and plant disease from field photographs then generation of context-aware herbicide recommendations (diagnosis -> severity estimate -> treatment options)

1. to run filter_datasets.py

- Maps various folder names and labels from different sources into 10 standard tomato disease classes.
- Remove duplicates
- Standardise image size and split inot `train`, `val` and `test` folders

```
python filter_datasets.py \
    --plantvillage data/PlantVillage \
    --plantdoc     data/PlantDoc \
    --plantseg     data/plantseg \
    --output       data/processed \
    --size         256
```
2. to run data_loader.py (data augmentation + loader)

- add changes to training images to make the model more robust
- \+ class balancing + efficiency

```
python filter_datasets.py \
    --data data/processed --size 224 --batch 32 --preview
```

3. to run classifier.py

This script defines the "brain" of the operation. It uses transfer learning, taking models already trained on millions of images (ImageNet) and adapting them for leaves.

```
python src/classifier.py 
    --backbone efficientnet --num-classes 10 --summary --test-forward
```

=> train.py will then call functios from both scripts to bridge the gap between the data and the model
