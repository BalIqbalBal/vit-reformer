import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcMarginProduct, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, x, labels):
        # Normalize features and weights
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # Add margin to target class
        theta = torch.acos(cosine.clamp(-1.0, 1.0))
        target_logits = torch.cos(theta + self.m)
        # One-hot encoding for labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        # Adjust logits
        logits = self.s * (one_hot * target_logits + (1.0 - one_hot) * cosine)
        return logits

class FaceTransformer(nn.Module):
    def __init__(self, num_classes, arc_s=30.0, arc_m=0.50):
        super(FaceTransformer, self).__init__()
        self.backbone = create_model('vit_base_patch16_224', pretrained=False)
        self.backbone.head = nn.Identity()
        self.arcface = ArcMarginProduct(in_features=self.backbone.embed_dim, out_features=num_classes, s=arc_s, m=arc_m=0.50)

    def forward(self, x, labels):
        features = self.backbone(x)
        logits = self.arcface(features, labels)
        return logits

    def extract_features(self, x):
        """
        Extract normalized feature embeddings for visualization.
        """
        return F.normalize(self.backbone(x), dim=1)