import torch
from torch import nn
from src.dataset import Vocabulary
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


class PreTrainedResNet50(nn.Module):
    def __init__(self, dropout_rate:float, embed_size:int = 512, fine_tune:bool=False):
      super().__init__()

      self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

      # Freezing all layers from the model, except the classifier
      for param in self.model.parameters():
        param.requires_grad=False
      for param in self.model.fc.parameters():
        param.requires_grad=True

      # Replacing the classifier layer with the adjusted with embed size as the output dimension
      self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=embed_size, bias=True)
      
      if fine_tune:
        # fine tunes the last convolutional layer and the last 2 Inverted Residual Blocks
        for param in self.model.layer4.parameters():
            param.requires_grad = True

      for module in self.model.modules():
          if isinstance(module, nn.BatchNorm2d):
                module.eval()
            
    def forward(self, images):
      return self.model(images)
  
class ScratchRNN(nn.Module):
  def __init__(self, embed_size,
               num_layers, hidden_size, dropout_rate, vocab, is_gru=True, pretrained_embeddings=None):

    super().__init__()

    self.end_idx = vocab.stoi["<EOS>"]
    self.pad_idx = vocab.stoi["<PAD>"]
    self.start_idx = vocab.stoi["<SOS>"]
    self.vocab_size = len(vocab)
    self.embed_size = embed_size

    if pretrained_embeddings is None:
      self.embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=self.pad_idx)
    else:
      self.embedding = nn.Embedding.from_pretrained(
        pretrained_embeddings,
        padding_idx=self.pad_idx,
        
      )

    if is_gru:
      self.rnn = nn.GRU(
          input_size=embed_size,
          hidden_size=hidden_size,
          num_layers=num_layers,
          batch_first=True,
          #dropout=dropout_rate,
          bidirectional=False 
      )
    else:
      self.rnn = nn.LSTM(
          input_size=embed_size,
          hidden_size=hidden_size,
          num_layers=num_layers,
          batch_first=True,
          #dropout=dropout_rate,
          bidirectional=False 
      )
    self.classifier = nn.Linear(hidden_size, self.vocab_size)
    self.dropout = nn.Dropout(dropout_rate)

  def forward(self, encoded_images, captions, captions_lengths):
    embedded_captions = self.dropout(self.embedding(captions))

    # Placing the image vector as the first token of the sequence
    embedded_captions = torch.cat((encoded_images.unsqueeze(1), embedded_captions), dim=1) # encoded image shape: (batch, embed dim) / embedded captions shape: (batch, seq len, embed dim)
    captions_lengths = captions_lengths + 1

    # This line improves EFFICIENCY and CORRECTNESS --> It guarantees that the RNN only processes the real tokens, ignoring the padding tokens
    packed_embedded_captions = nn.utils.rnn.pack_padded_sequence(
        embedded_captions, captions_lengths, batch_first=True, enforce_sorted=False
    )

    hiddens, _ = self.rnn(packed_embedded_captions)

    # Since the input was a PackedSequence, the output will also be, so we need to unpack it
    hiddens_tensor, _ = nn.utils.rnn.pad_packed_sequence(hiddens, batch_first=True) # hiddens_tensor shape: (batch, seq len, embed_dim)

    outputs = self.classifier(hiddens_tensor) # outputs shape: (batch, seq len, vocab size)

    # Returns raw logits for each word in the vocabulary
    return outputs

class ImageCaptionModel(nn.Module):
  def __init__(self, cnn:PreTrainedMobileNetV3, rnn:ScratchRNN):
    super().__init__()
    self.cnn = cnn
    self.rnn = rnn

  def forward(self, images, captions, captions_lengths):
    logits = self.rnn(self.cnn(images), captions, captions_lengths)
    return logits

  def infer(self, image, max_caption_len:int=100):
    done = False
    predicted_sequence = []
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