import torch
import numpy as np
from torch.utils.data import Dataset
from collections import Counter

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import os

# Tokens especiais:
# - PAD: Preenchimento (para que todas as frases no batch tenham o mesmo tamanho).
# - SOS: Start of Sentence (início da frase).
# - EOS: End of Sentence (fim da frase).
# - UNK: Unknown (para palavras que aparecem pouco e não entrarão no vocabulário)

# Dicionários:
# - itos: index para string 
# - stoi: string para index 

class Vocabulary:
    def __init__(self, min_freq=1):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.min_freq = min_freq

    def __len__(self): return len(self.itos)

    def split_tokens(self, text):
        return [word.lower() for word in text.split()]

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in self.split_tokens(sentence):
                frequencies[word] += 1
                if frequencies[word] == self.min_freq:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def tokenize(self, text):
        tokenized_text = self.split_tokens(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]


class FlickrDataset(Dataset):
    def __init__(self, root_dir, df, vocab, transform=None, max_tokens=100, is_eval=False):
        self.root_dir = root_dir
        self.df = df
        self.vocab = vocab
        self.transform = transform
        self.max_tokens = max_tokens
        self.is_eval = is_eval

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_id = self.df.iloc[index]['image']
        img_path = os.path.join(self.root_dir, img_id)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        if self.is_eval:
            captions = self.df.iloc[index]['caption_clean']
            return image, captions

        caption = self.df.iloc[index]['caption_clean']

        tokens = self.vocab.tokenize(caption)

        tokenized_caption = [self.vocab.stoi["<SOS>"]] + tokens + [self.vocab.stoi["<EOS>"]]

        # Truncamento (Corta se exceder max_tokens)
        if len(tokenized_caption) > self.max_tokens:
            tokenized_caption = tokenized_caption[:self.max_tokens]

        return image, torch.tensor(tokenized_caption)

    
def collate_fn(batch, padding_idx=0):
    # Sorting batch in descending order here to save some computational cost instead of sorting the batch in the forward pass by the pack padded sequence method
    batch.sort(key=lambda x : len(x[1]), reverse=True)
    images = torch.stack([x[0] for x in batch], dim=0)
    tokenized_captions = [x[1] for x in batch]
    tokenized_captions_lengths = torch.tensor([len(x[1]) for x in batch], dtype=torch.long)
    tokenized_captions = pad_sequence(tokenized_captions, batch_first=True, padding_value=padding_idx)
    return images, tokenized_captions, tokenized_captions_lengths

def preprocess_data(train_dataset: FlickrDataset, val_dataset: FlickrDataset, test_dataset: FlickrDataset, batch_size:int, padding_idx=0):

    cpu_count = int(os.cpu_count() or 2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=cpu_count, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=cpu_count, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=cpu_count, pin_memory=True)

    return train_loader, val_loader, test_loader
    
def build_weights_matrix(vocab, pretrained_embeddings_path, embedding_dim=300):
    embedding_index = {}
    with open (pretrained_embeddings_path,'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:],dtype='float32')
            embedding_index[word] = coefs

    vocab_size = len(vocab)
    weight_matrix = np.random.normal(scale=0.6, size=(vocab_size, embedding_dim))

    weight_matrix[vocab.stoi["<PAD>"]] = np.zeros((embedding_dim,))

    words_found = 0
    words_not_found = []
    
    for i in range(vocab_size):
        word = vocab.itos[i]
        if word in embedding_index:
            weight_matrix[i] = embedding_index[word]
            words_found += 1
        else:
            words_not_found.append(word)
    
    print(f"Sucesso: {words_found}/{vocab_size} palavras encontradas nos embeddings pré-treinados")

    return torch.from_numpy(weight_matrix).float(), words_not_found