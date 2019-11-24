# Neuroclass
A integrated and intuitive brain deformation and system healthiness classifier

The model(based off of the famous Resnet) is in this repo. The model used the publicly available NIH malaria dataset on rescaled malaria cell pictures. Using this model, we get arounda 96% accuracy rate in detecting whether or not a cell is infected or not. 

The file 'classifier.h5' contains the full h5 file of the resnet model including all weights biases, and values. To run the model, simple load the model and apply the predict function onto an image for a probabalistic distribution of whether or not a cell is infected. 

The web app is a cleverly integrated system that uses flask to allow a user to upload an image and have it be classified by the model. 
