
import torch
from torch import nn
import torchvision.models as models

class ScratchGRU(nn.Module):
  def __init__(self, embed_size, vocab_size, pad_idx, end_idx,
               num_layers, hidden_size, dropout_rate):

    super().__init__()

    self.end_idx = end_idx

    self.embed_size = embed_size

    self.embedding = nn.Embedding(vocab_size, self.embed_size, padding_idx=pad_idx)

    self.gru = nn.GRU(
        input_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        dropout=dropout_rate,
        bidirectional=False 
    )

    self.classifier = nn.Linear(hidden_size, vocab_size)

    self.dropout = nn.Dropout(dropout_rate)

  def forward(self, encoded_images, captions, captions_lengths):
    embedded_captions = self.dropout(self.embedding(captions))

    # Placing the image vector as the first token of the sequence
    embedded_captions = torch.cat((encoded_images.unsqueeze(1), embedded_captions), dim=1) # encoded image shape: (batch, embed dim) / embedded captions shape: (batch, seq len, embed dim)
    captions_lengths = captions_lengths + 1

    # This line improves EFFICIENCY and CORRECTNESS --> It guarantees that the RNN only processes the real tokens, ignoring the padding tokens
    packed_embedded_captions = nn.utils.rnn.pack_padded_sequence(
        embedded_captions, captions_lengths.cpu(), batch_first=True, enforce_sorted=False
    )

    hiddens, _ = self.gru(packed_embedded_captions)

    # Since the input was a PackedSequence, the output will also be, so we need to unpack it
    hiddens_tensor, _ = nn.utils.rnn.pad_packed_sequence(hiddens, batch_first=True) # hiddens_tensor shape: (batch, seq len, embed_dim)

    outputs = self.classifier(hiddens_tensor) # outputs shape: (batch, seq len, vocab size)

    # Returns raw logits for each word in the vocabulary
    return outputs