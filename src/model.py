
# src/model.py
import timm
import torch.nn as nn

def create_model(num_classes=5):
    # Load the same architecture you trained with
    model = timm.create_model('tf_efficientnet_b0.ns_jft_in1k', pretrained=False)
    
    # Change the final layer to match your number of classes
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    

    return model
