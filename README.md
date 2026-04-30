# Crop-Disease-and-Treatment-Advisor
Identification of plant and plant disease from field photographs then generation of context-aware herbicide recommendations (diagnosis -> severity estimate -> treatment options)

1. to run filter_datasets.py
```
python filter_datasets.py \
    --plantvillage data/PlantVillage \
    --plantdoc     data/PlantDoc \
    --plantseg     data/plantseg \
    --output       data/processed \
    --size         256
```
2. to run data_loader.py (data augmentation + loader)
```
python filter_datasets.py \
    --data data/processed --size 224 --batch 32 --preview
```

3. to run classifier.py
```
python src/classifier.py 
    --backbone efficientnet --num-classes 10 --summary --test-forward
```
