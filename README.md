# Sorghum100-CultivarId
Code, Report and Presentation of my solution to the Sorghum -100 Cultivar Identification - FGVC 9 Kaggle Competition

<img src="figures/data_augmetation.png" alt="Italian Trulli">

# Before running

Must download data from https://www.kaggle.com/competitions/sorghum-id-fgvc-9/data
Put data in a folder named "data".
Create empty folders "embeddings", "models", "plots" and "Submissions" for the rest of the experiments.

# Explanation of all files and folders

Folders:
- Plots: stores the train and val accuracies of the experiments that were done
- models: stores the models ready to be loaded by pytorch
- figures: Are some figures we saved
- Embeddings: store the embeddings obtained by using the file create_embeddings.py
- Submissions: Here we store the csv submissions for the kaggle competition
- data: Here we store the competition data

Files:

- TrainModel.py: With this file we trained most of our models
- utils.py: some useful functions
- create_embeddings.py: How we create the embeddings
- CreateSubmissions.ipynb: Notebook to create submissions, in order we have the vanilla inference, the ensembling with averaging, and the ensembling with maxes (using the mode function).
- PlotMaker.ipynb: To do our plots
- WeightViz.ipynb: Extracting some weights
- SorghumDataset.py: Implementation of the class of our data.
- VisualizingDatasetTransforms.ipynb: See effects of transforms on data

# Best Config
Use the following transformations
```python
trans = [
        transforms.RandomChoice([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.RandomPerspective()

        ]),
        transforms.RandomChoice([
            transforms.RandomGrayscale(p=0.1),
            transforms.ColorJitter(brightness=.3, hue=.4),
            transforms.GaussianBlur(kernel_size=5),
            transforms.RandomInvert(p=0.1)
        ]),

        transforms.RandomChoice([
            transforms.RandomResizedCrop(size=(512, 512), scale=(0.1, 1.00)),
            transforms.Resize(size=(512, 512)),
        ]),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(inplace=True),
    ]
```    
    
Training a EfficientNet-b4 backbone with pretrained weights, learning rate 1e-3 and 30 epochs

#Should give around 77% accuracy on inference, and 80% with ensembling
