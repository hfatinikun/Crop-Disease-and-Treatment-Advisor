# Crop-Disease-and-Treatment-Advisor
Identification of plant and plant disease from field photographs then generation of context-aware herbicide recommendations (diagnosis -> severity estimate -> treatment options)

to run filter_datasets.py
```
python filter_datasets.py \
    --plantvillage ../data/PlantVillage \
    --plantdoc     ../data/PlantDoc \
    --plantseg     ../data/plantseg \
    --output       ../data/processed \
    --size         256
```
