# Sorghum100-CultivarId
Code, Report and Presentation of my solution to the Sorghum -100 Cultivar Identification - FGVC 9 Kaggle Competition


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
- TrainModel_cleaned.py: Here the files were reorganized and better distributed among utils.py and SorghumDataset.py. The file works, however due to the way that consoles, and kernels work in Azure (you have to close and reopen consoles for some changes to be taken into account), it was decided to just use the above file for training. This file is more readable so you might use it as reference.
- utils.py: some useful functions
- create_embeddings.py: How we create the embeddings
- CreateSubmissions.ipynb: Notebook to create submissions, in order we have the vanilla inference, the ensembling with averaging, and the ensembling with maxes (using the mode function).
- PlotMaker.ipynb: To do our plots
- WeightViz.ipynb: Extracting some weights
- SorghumDataset.py: Implementation of the class of our data.
- VisualizingDatasetTransforms.ipynb: See effects of transforms on data
