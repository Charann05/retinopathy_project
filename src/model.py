
# src/model.py
import timm
import torch.nn as nn

def create_model(n_classes=5, model_name='tf_efficientnet_b0.ns_jft_in1k', pretrained=True):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=n_classes)
    return model