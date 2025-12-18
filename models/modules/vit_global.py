import torch
import torch.nn as nn
import torch.nn.functional as F
import models.vision_transformer as vision_transformer


class ViTGlobalModel(nn.Module):
    """
    只用ViT全局特征的模型
    不使用RoI pooling，只用[CLS] token
    """
    def __init__(self, predicate_dim, hidden_dim=512, backbone="vit_base", 
                 pretrain_ckp="", freeze_backbone=True, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.predicate_dim = predicate_dim
        
        # 加载ViT backbone
        self.backbone = vision_transformer.__dict__[backbone](pretrain_ckp=pretrain_ckp)
        backbone_dim = self.backbone.embed_dim  # 768 for ViT-Base
        
        # 是否冻结backbone
        if freeze_backbone:
            print("\n" + "="*70)
            print("🧊 Freezing ViT backbone in ViTGlobalModel...")
            print("="*70)
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False
                print(f"  ❄️  Frozen: backbone.{name}")
            
            frozen_params = sum(p.numel() for p in self.backbone.parameters())
            print(f"✅ Successfully frozen {frozen_params:,} parameters in ViT backbone")
            print("="*70 + "\n")
        
        # Bbox特征编码器（将4维bbox映射到高维）
        bbox_embed_dim = 128
        self.bbox_encoder = nn.Sequential(
            nn.Linear(4, bbox_embed_dim),
            nn.ReLU(),
            nn.LayerNorm(bbox_embed_dim),
            nn.Linear(bbox_embed_dim, bbox_embed_dim),
            nn.ReLU(),
            nn.LayerNorm(bbox_embed_dim)
        )
        
        # 特征融合层
        # 输入: [CLS] token (768) + subject_bbox_emb (128) + object_bbox_emb (128)
        fusion_input_dim = backbone_dim + bbox_embed_dim * 2
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3)
        )
        
        # 关系分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, predicate_dim)
        )
    
    def forward(self, full_im, bbox_s, bbox_o, predicate):
        """
        Args:
            full_im: (B, 3, H, W)
            bbox_s: (B, 4) - subject bbox, normalized
            bbox_o: (B, 4) - object bbox, normalized
            predicate: (B,) - predicate index
        """
        # 1. 提取ViT全局特征 ([CLS] token)
        vit_features = self.backbone(full_im)  # (B, 197, 768) for ViT-Base 224x224
        cls_token = vit_features[:, 0, :]  # (B, 768) - 只取[CLS] token
        
        # 2. 编码bbox信息
        bbox_s_emb = self.bbox_encoder(bbox_s)  # (B, 128)
        bbox_o_emb = self.bbox_encoder(bbox_o)  # (B, 128)
        
        # 3. 融合所有特征
        combined = torch.cat([cls_token, bbox_s_emb, bbox_o_emb], dim=1)  # (B, 1024)
        fused_feature = self.fusion(combined)  # (B, hidden_dim)
        
        # 4. 分类
        logits = self.classifier(fused_feature)  # (B, predicate_dim)
        
        # 5. 选择对应predicate的输出
        predi_onehot = F.one_hot(predicate, num_classes=self.predicate_dim).float()
        output = torch.sum(logits * predi_onehot, dim=1)  # (B,)
        
        return output


class ViTBboxAttentionModel(nn.Module):
    """
    使用bbox位置信息引导注意力
    软性地聚焦到相关区域
    """
    def __init__(self, predicate_dim, hidden_dim=512, backbone="vit_base",
                 pretrain_ckp="", freeze_backbone=True, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.predicate_dim = predicate_dim
        
        self.backbone = vision_transformer.__dict__[backbone](pretrain_ckp=pretrain_ckp)
        backbone_dim = self.backbone.embed_dim
        
        if freeze_backbone:
            print("\n" + "="*70)
            print("🧊 Freezing ViT backbone in ViTBboxAttentionModel...")
            print("="*70)
            for param in self.backbone.parameters():
                param.requires_grad = False
            frozen_params = sum(p.numel() for p in self.backbone.parameters())
            print(f"✅ Successfully frozen {frozen_params:,} parameters")
            print("="*70 + "\n")
        
        # ViT的patch数量 (14x14 for 224x224 input with patch_size=16)
        self.num_patches_per_side = 14
        self.num_patches = self.num_patches_per_side ** 2
        
        # 位置编码生成器
        self.position_encoder = nn.Sequential(
            nn.Linear(2, 64),  # (x, y) -> 64
            nn.ReLU(),
            nn.Linear(64, 1)   # 64 -> 1 (attention score)
        )
        
        # Bbox编码器
        bbox_embed_dim = 128
        self.bbox_encoder = nn.Sequential(
            nn.Linear(4, bbox_embed_dim),
            nn.ReLU(),
            nn.LayerNorm(bbox_embed_dim)
        )
        
        # 特征融合
        # subject_feature + object_feature + bbox_features
        fusion_input_dim = backbone_dim * 2 + bbox_embed_dim * 2
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, predicate_dim)
        )
    
    def get_patch_positions(self, device):
        """
        获取每个patch的中心位置
        Returns: (num_patches, 2) - 归一化的(x, y)坐标
        """
        positions = []
        for i in range(self.num_patches_per_side):
            for j in range(self.num_patches_per_side):
                # patch中心位置（归一化到[0,1]）
                x = (i + 0.5) / self.num_patches_per_side
                y = (j + 0.5) / self.num_patches_per_side
                positions.append([x, y])
        
        return torch.tensor(positions, dtype=torch.float32, device=device)
    
    def compute_bbox_attention(self, bbox, patch_positions):
        """
        计算bbox对每个patch的注意力权重
        Args:
            bbox: (B, 4) - (x1, x2, y1, y2)
            patch_positions: (num_patches, 2) - (x, y)
        Returns:
            attention_weights: (B, num_patches)
        """
        B = bbox.shape[0]
        num_patches = patch_positions.shape[0]
        
        # 扩展维度
        bbox = bbox.unsqueeze(1).expand(B, num_patches, 4)  # (B, 196, 4)
        patch_pos = patch_positions.unsqueeze(0).expand(B, num_patches, 2)  # (B, 196, 2)
        
        # 计算每个patch是否在bbox内
        x, y = patch_pos[:, :, 0], patch_pos[:, :, 1]
        x1, x2, y1, y2 = bbox[:, :, 0], bbox[:, :, 1], bbox[:, :, 2], bbox[:, :, 3]
        
        # 在bbox内的patch得到更高权重
        inside = ((x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)).float()
        
        # 计算距离bbox中心的距离
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        dist = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # 位置编码生成attention
        pos_encoding = self.position_encoder(patch_pos)  # (B, 196, 1)
        pos_encoding = pos_encoding.squeeze(-1)  # (B, 196)
        
        # 组合：在bbox内 + 距离 + 位置编码
        attention_logits = inside * 2.0 - dist + pos_encoding
        attention_weights = F.softmax(attention_logits, dim=1)  # (B, 196)
        
        return attention_weights
    
    def forward(self, full_im, bbox_s, bbox_o, predicate):
        # 1. 提取ViT特征
        vit_features = self.backbone(full_im)  # (B, 197, 768)
        patch_features = vit_features[:, 1:, :]  # (B, 196, 768) - 去掉[CLS]
        
        # 2. 获取patch位置
        patch_positions = self.get_patch_positions(full_im.device)  # (196, 2)
        
        # 3. 计算注意力权重
        attn_s = self.compute_bbox_attention(bbox_s, patch_positions)  # (B, 196)
        attn_o = self.compute_bbox_attention(bbox_o, patch_positions)  # (B, 196)
        
        # 4. 加权聚合特征
        subject_feature = torch.sum(
            patch_features * attn_s.unsqueeze(-1),  # (B, 196, 768)
            dim=1
        )  # (B, 768)
        
        object_feature = torch.sum(
            patch_features * attn_o.unsqueeze(-1),
            dim=1
        )  # (B, 768)
        
        # 5. Bbox编码
        bbox_s_emb = self.bbox_encoder(bbox_s)
        bbox_o_emb = self.bbox_encoder(bbox_o)
        
        # 6. 融合
        combined = torch.cat([
            subject_feature, object_feature,
            bbox_s_emb, bbox_o_emb
        ], dim=1)
        
        fused_feature = self.fusion(combined)
        
        # 7. 分类
        logits = self.classifier(fused_feature)
        predi_onehot = F.one_hot(predicate, num_classes=self.predicate_dim).float()
        output = torch.sum(logits * predi_onehot, dim=1)
        
        return output
