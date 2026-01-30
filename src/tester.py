import torch
from torch import nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor


class ImageCaptionTester:
    def __init__(self, model, device, vocab):
        """
        Initializes the tester with the model, execution device, and vocabulary.
        """
        self.model = model
        self.device = device
        self.vocab = vocab
        
        # Initialize evaluation metrics
        # Bleu(4) calculates Bleu-1, Bleu-2, Bleu-3, and Bleu-4
        self.bleu = Bleu(4)
        self.cider = Cider()
        self.meteor = Meteor()

    def test(self, dataloader):
        """
        Runs the evaluation loop over the provided test dataloader.
        """
        self.model.eval()
        gts = {} # Ground Truth dictionary
        res = {} # Results (Predictions) dictionary
        
        # Progress bar for visual feedback during evaluation
        pbar = tqdm(dataloader, desc="Evaluating", leave=True)
        
        with torch.no_grad():
            for i, (images, captions_tuple) in enumerate(pbar):
                # Move images to the designated device (GPU/CPU)
                images = images.to(self.device)
                
                # The model generates token IDs based on the visual features
                predicted_indices = self.model.infer(images, max_caption_len=50)
                
                # Mapping IDs back to words and filtering out special tokens (<SOS>, <EOS>, <PAD>)
                predicted_caption = " ".join([
                    self.vocab.itos[idx] for idx in predicted_indices 
                    if idx not in [self.vocab.stoi["<SOS>"], self.vocab.stoi["<EOS>"], self.vocab.stoi["<PAD>"]]
                ])
                
                # Image ID must be a string to be compatible with pycocoevalcap
                img_id = str(i)
                
                # The prediction must be a list containing a single string
                res[img_id] = [predicted_caption]
                
                # Since the DataLoader groups multiple captions into tuples/lists, 
                # we ensure they are flattened into a simple list of strings.
                if isinstance(captions_tuple, (list, tuple)):
                    # Extract string from nested structure if necessary
                    gts[img_id] = [str(c[0]) if isinstance(c, (list, tuple)) else str(c) for c in captions_tuple]
                else:
                    gts[img_id] = [str(captions_tuple)]

        # 4. COMPUTE AND RETURN ALL METRICS
        return self.metrics_measures(res, gts)

    def metrics_measures(self, preds, gt):
        """
        Computes scores for Bleu, Cider, and Meteor based on predictions and ground truths.
        """
        metrics = {}
        
        # Compute Bleu scores (returns a list of 4 floats)
        score_bleu, _ = self.bleu.compute_score(gt, preds)
        for n in range(4):
            metrics[f"Bleu_{n+1}"] = score_bleu[n]
            
        # Compute Meteor score
        score_meteor, _ = self.meteor.compute_score(gt, preds)
        metrics["Meteor"] = score_meteor
        
        # Compute Cider score
        score_cider, _ = self.cider.compute_score(gt, preds)
        metrics["Cider"] = score_cider
        
        return metrics
    
    def show_example(self, dataloader, image_idx):
        self.model.eval()
        iter_dataloader = iter(dataloader)
        
        for _ in range(image_idx):
            image, _ = next(iter_dataloader)

        predicted_indices = self.model.infer(image, max_caption_len=50)
        
        predicted_caption = " ".join([
            self.vocab.itos[idx] for idx in predicted_indices 
            if idx not in [self.vocab.stoi["<SOS>"], self.vocab.stoi["<EOS>"], self.vocab.stoi["<PAD>"]]
        ])
        print(image.shape)
        image = (torch.squeeze(image))
        image = image.permute(1, 2, 0)
        print(image.shape)
        plt.title(predicted_caption)
        plt.imshow(image)

if __name__ == "__main__":

    tester = ImageCaptionTester(model = None, device = "cpu")
    pred = {"1" : ["oi tudo certo"]}
    gt = {"1": ["oi tudo", "oi tudo bem", "oi tudo certo", "oi", "tudo certo"]}
    result = tester.metrics_measures(pred, gt)
    print(result)
