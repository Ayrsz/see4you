import torch
from torch import nn
import torchvision.models as models

class ImageCaptioningModel(nn.Module):
  def __init__(self, cnn:nn.Module, rnn:nn.Module):
    super().__init__()
    self.cnn = cnn
    self.rnn = rnn

  def forward(self, images, captions, captions_lengths):
    logits = self.rnn(self.cnn(images), captions, captions_lengths)
    return logits

  def infer(self, image, vocabulary, max_caption_len:int=50):
    done = False
    predicted_sequence = []
    with torch.no_grad():
      # Check the images dimensions
      if len(image.shape) == 3:
        image = image.unsqueeze(0)  # Add batch dimension to the image
      encoded = self.cnn(image).unsqueeze(1) # Add seq len dimension to the encoded image of shape (1, embed_size) 
      h_i = None
      while not done:
        hiddens, h_i = self.rnn.rnn(encoded, h_i)
        outputs = self.rnn.classifier(hiddens)

        predicted_token = torch.argmax(outputs, dim=2)
        predicted_sequence.append(predicted_token.item())

        # Ends the loop when the networks predicts the <END> token or when the captions reaches the maximum length
        if(len(predicted_sequence) >= max_caption_len) or (predicted_token.item() == self.rnn.end_idx):
          done = True
          
        encoded = self.rnn.embedding(predicted_token)
        encoded = self.rnn.dropout(encoded)

    return predicted_sequence