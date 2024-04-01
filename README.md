This repo contains my competition notebooks sbumitted for the competition "Classifying Harmful Brain Activities" (organized by Harvard Medical School and hosted on Kaggle)

https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/overview

The list of notebooks included are:
   - HMS: Inception-Resnet-v2 5fold-CV (Training)
   - HMS: Inception-ResNet-v2 5 fold-CV (Inference)
   - HMS: MobileNetV2 5 fold CV (Training + Inference)
   - HMS: Ensemble of models (Submission notebook)
   - HMS: Ensemble of models- Version 2 (Submission notebook)
Utility files includes:
   - spec_eeg.py contains script to convert EGG data to spectrogram format.
   - hms_data_generator contains the custom data generator function
   - ResNet1d_GRU_hybrid.py contains all functions needed to run inference on a new model based on ResNet1d and GRU hybrid. 
