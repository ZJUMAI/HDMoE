import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from .util import initialize_weights
from .util import NystromAttention
from .util import BilinearFusion
from .util import SNN_Block
from .util import MultiheadAttention


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class Transformer_P(nn.Module):
    def __init__(self, feature_dim=512):
        super(Transformer_P, self).__init__()
        # Encoder
        self.pos_layer = PPEG(dim=feature_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        # Decoder

    def forward(self, features):
        # ---->pad
        H = features.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([features, features[:, :add_length, :]], dim=1)  # [B, N, 512]
        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)
        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]
        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]
        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]
        # ---->cls_token
        h = self.norm(h)
        return h[:, 0], h[:, 1:]


class Transformer_G(nn.Module):
    def __init__(self, feature_dim=512):
        super(Transformer_G, self).__init__()
        # Encoder
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        # Decoder

    def forward(self, features):
        # ---->pad
        cls_tokens = self.cls_token.expand(features.shape[0], -1, -1).cuda()
        h = torch.cat((cls_tokens, features), dim=1)
        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]
        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]
        # ---->cls_token
        h = self.norm(h)
        return h[:, 0], h[:, 1:]


class MoEGate(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.top_k = args.top_k  
        self.n_routed_experts = args.n_routed_experts

        self.scoring_func = 'softmax'  
        self.alpha = 0.01  
        self.seq_aux = False  

        # topk selection algorithm
        self.norm_topk_prob = False  
        self.gating_dim = args.gate_dim  
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init  as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        # compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        # select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum()  # * self.alpha
        else:
            aux_loss = 0
        return topk_idx, topk_weight, aux_loss, scores


class MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, args, out_dim):
        super().__init__()
        self.args = args
        self.out_dim = out_dim
        self.num_experts_per_tok = args.top_k  
        self.experts = nn.ModuleList(
            [expert(input_dim=args.gate_dim, hidden_dim=128, output_dim=out_dim, dropout_rate=0.1) for i in
             range(args.n_routed_experts)])  # hidden_dim=128, output_dim=64
        self.gate = MoEGate(args)
        if args.n_shared_experts is True:
            # intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = expert(input_dim=args.gate_dim, hidden_dim=128, output_dim=out_dim, dropout_rate=0.1)
            # hidden_dim=128, output_dim=64

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss, scores = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0) 
        y = torch.empty_like(hidden_states[:,:self.out_dim])

        for i, expert in enumerate(self.experts):
            y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])

        y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).squeeze(1)  # 8,64

        if y.shape[1] == 2:
            y = y.view(orig_shape[1], -1).view(1, -1)  # 1,1024
        else:
            y = y.view(1,*y.shape)#*orig_shape

        y = torch.concat([y.view(1, -1), self.shared_experts(identity).view(1, -1)], dim=0)  # (2,256)
        return y, topk_weight, aux_loss, scores


class expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(expert, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        return x

class HDMoE(nn.Module):
    def __init__(self, args, omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4, fusion="concat",
                 model_size="small"):
        super(HDMoE, self).__init__()
        self.args = args
        self.seg_length = [1, 2, 4, 8, 16, 32, 64, 128]
        self.seg_length2 = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.fusion = fusion
        ###
        self.size_dict = {
            "pathomics": {"small": [1024, 256, 256], "large": [1024, 512, 256]},
            "genomics": {"small": [1024, 256], "large": [1024, 1024, 1024, 256]},
        }
        # Pathomics Embedding Network
        hidden = self.size_dict["pathomics"][model_size]
        fc = []
        for idx in range(len(hidden) - 1):
            fc.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(0.25))
        self.pathomics_fc = nn.Sequential(*fc)
        # Genomic Embedding Network
        hidden = self.size_dict["genomics"][model_size]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.genomics_fc = nn.ModuleList(sig_networks)

        # Encoder
        self.pathomics_encoder = Transformer_P(feature_dim=hidden[-1])
        self.genomics_encoder = Transformer_G(feature_dim=hidden[-1])

        # MoE Decouple
        self.pathomics_moe = MoE(args, out_dim=args.gate_dim)
        self.genomics_moe = MoE(args, out_dim=args.gate_dim)
        self.fusion_moe = MoE(args, out_dim=args.gate_dim//2)

        self.emb1 = nn.Linear(args.gate_dim * 2, args.gate_dim * 2)
        self.emb2 = nn.Linear(args.gate_dim * 2, args.gate_dim * 2)

        self.classifier = nn.Linear(hidden[-1] * 4, self.n_classes)
        self.apply(initialize_weights)

    def forward(self, **kwargs):
        # meta genomics and pathomics features
        x_path = kwargs["x_path"]
        x_omic = [kwargs["x_omic%d" % i] for i in range(1, 7)]

        # Enbedding
        # genomics embedding
        genomics_features = [self.genomics_fc[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)]
        genomics_features = torch.stack(genomics_features).unsqueeze(0) 
        # pathomics embedding
        pathomics_features = self.pathomics_fc(x_path).unsqueeze(0)

        # encoder
        # pathomics encoder
        cls_token_pathomics_encoder, patch_token_pathomics_encoder = self.pathomics_encoder(
            pathomics_features)  # cls token + patch tokens 
        # genomics encoder
        cls_token_genomics_encoder, patch_token_genomics_encoder = self.genomics_encoder(
            genomics_features)  # cls token + patch tokens

        cls_token_genomics_encoder = cls_token_genomics_encoder.view(1, -1, self.args.gate_dim)  
        genomics_feature, genomics_score, genomics_loss, genomics_scores = self.genomics_moe(cls_token_genomics_encoder) 

        cls_token_pathomics_encoder = cls_token_pathomics_encoder.view(1, -1, self.args.gate_dim)
        pathomics_feature, pathomics_score, pathomics_loss, pathomics_scores = self.pathomics_moe(cls_token_pathomics_encoder)  

        emb_specific = self.emb1(torch.concat([genomics_feature[:1,:],pathomics_feature[:1,:]],dim=1)) 
        emb_share = self.emb2(torch.concat([genomics_feature[1:,:],pathomics_feature[1:,:]],dim=1)) 

        cat_total = torch.concat([genomics_feature, pathomics_feature], dim=-1)  

        seg_l = random.choice(self.seg_length)
        cat_fusion = cat_total.reshape(4, seg_l, self.args.gate_dim // seg_l).transpose(1, 0).reshape(1, -1).view(1, -1, self.args.gate_dim)
        fusion_feature, fusion_score, fusion_loss, fusion_scores = self.fusion_moe(cat_fusion) 
        seg_l2 = random.choice(self.seg_length2)
        out = fusion_feature.reshape(2, seg_l2, (self.args.gate_dim * 2) // seg_l2).transpose(1, 0).reshape(1, -1)  

        # predict
        logits = self.classifier(out)  # [1, n_classes]

        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return [genomics_feature, pathomics_feature, fusion_feature, emb_specific, emb_share], \
               [genomics_scores,pathomics_scores,fusion_scores], [genomics_loss, pathomics_loss,fusion_loss],\
               hazards, S, logits
