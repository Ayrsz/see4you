import torch
from torch import nn
import torchvision.models as models

class PreTrainedMobileNetV3(nn.Module):
  def __init__(self, dropout_rate:float, embed_size:int = 512, fine_tune:bool=False):
    super().__init__()

    self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)

    # Freezing all layers from the model, except the classifier
    for param in self.model.features.parameters():
      param.requires_grad=False
    for param in self.model.avgpool.parameters():
      param.requires_grad=False

    # Replacing the classifier layer with the adjusted with embed size as the output dimension
    self.model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(in_features=self.model.features[-1].out_channels, out_features=embed_size, bias=True)
    )

    if fine_tune:
      # fine tunes the last convolutional layer and the last 2 Inverted Residual Blocks
      for feature in self.model.features[-3:]:
        for param in feature.parameters():
          param.requires_grad = True
          
  def forward(self, images):
    return self.model(images)