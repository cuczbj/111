import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import build_model
from dataset import SpatialSenseDataset
from configs import get_cfg_defaults


def bbox_overlap(bbox1, bbox2):
    """计算两个bbox的IoU"""
    y0_1, y1_1, x0_1, x1_1 = bbox1
    y0_2, y1_2, x0_2, x1_2 = bbox2
    
    y0_inter = max(y0_1, y0_2)
    y1_inter = min(y1_1, y1_2)
    x0_inter = max(x0_1, x0_2)
    x1_inter = min(x1_1, x1_2)
    
    if y1_inter <= y0_inter or x1_inter <= x0_inter:
        return 0.0
    
    inter_area = (y1_inter - y0_inter) * (x1_inter - x0_inter)
    bbox1_area = (y1_1 - y0_1) * (x1_1 - x0_1)
    bbox2_area = (y1_2 - y0_2) * (x1_2 - x0_2)
    
    union_area = bbox1_area + bbox2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    
    return iou


class SpatialSenseDatasetOriginal(SpatialSenseDataset):
    """不涂红的数据集版本"""
    def __getitem__(self, idx, visualize=False):
        result = super().__getitem__(idx, visualize)
        return result


class SpatialSenseDatasetRedObject(SpatialSenseDataset):
    """涂红object框的数据集版本"""
    def __getitem__(self, idx, visualize=False):
        annot = self.annotations[idx]
        
        t_s = self._getT(annot["subject"]["bbox"], annot["object"]["bbox"])
        t_o = self._getT(annot["object"]["bbox"], annot["subject"]["bbox"])

        ys0, ys1, xs0, xs1 = annot["subject"]["bbox"]
        yo0, yo1, xo0, xo1 = annot["object"]["bbox"]
        union_bbox = self._getUnionBBox(annot["subject"]["bbox"], annot["object"]["bbox"],
                                        annot["height"], annot["width"])

        datum = {
            "url": annot["url"],
            "_id": annot["_id"],
            "subject": {
                "name": annot["subject"]["name"],
                "idx": self.objects.index(annot['subject']['name']),
                "bbox": np.asarray(
                    [
                        ys0 / annot["height"],
                        ys1 / annot["height"],
                        xs0 / annot["width"],
                        xs1 / annot["width"],
                    ],
                    dtype=np.float32,
                ),
                "t": np.asarray(t_s, dtype=np.float32),
            },
            "object": {
                "name": annot["object"]["name"],
                "idx": self.objects.index(annot['subject']['name']),
                "bbox": np.asarray(
                    [
                        yo0 / annot["height"],
                        yo1 / annot["height"],
                        xo0 / annot["width"],
                        xo1 / annot["width"],
                    ],
                    dtype=np.float32,
                ),
                "t": np.asarray(t_o, dtype=np.float32),
            },
            "label": annot["label"],
            "predicate": {
                'name': annot["predicate"],
                'idx': self.predicates.index(annot["predicate"]),
                'bbox': np.asarray(
                    [
                        union_bbox[0] / annot["height"],
                        union_bbox[1] / annot["height"],
                        union_bbox[2] / annot["width"],
                        union_bbox[3] / annot["width"],
                    ], dtype=np.float32,
                )
            },
            'rgb_source': self.get_img_path(annot["url"], self.img_path),
        }

        if self.load_image:
            import torchvision.transforms.functional as TF
            import torchvision.transforms as transforms
            from PIL import Image
            import cv2
            
            img = self.read_img(annot["url"], self.img_path)
            ih, iw = img.shape[:2]

            # ===== 将object框内涂成红色 =====
            yo0, yo1, xo0, xo1 = annot["object"]["bbox"]
            img[yo0:yo1, xo0:xo1, :] = [255.0, 0.0, 0.0]

            bbox_mask = np.stack(
                [
                    self._getDualMask(ih, iw, annot["subject"]["bbox"], ih, iw).astype(np.uint8),
                    self._getDualMask(ih, iw, annot["object"]["bbox"], ih, iw).astype(np.uint8),
                    np.zeros((ih, iw), dtype=np.uint8),
                ],
                axis=2,
            )

            if self.crop:
                enlarged_union_bbox = self.enlarge(union_bbox, 1.25, ih, iw, )
                full_img = Image.fromarray(img[enlarged_union_bbox[0]:enlarged_union_bbox[1], enlarged_union_bbox[2]:enlarged_union_bbox[3], :].astype(np.uint8, copy=False), mode="RGB")
                bbox_mask = Image.fromarray(bbox_mask[enlarged_union_bbox[0]:enlarged_union_bbox[1], enlarged_union_bbox[2]:enlarged_union_bbox[3], :].astype(np.uint8, copy=False), mode="RGB")
            else:
                full_img = Image.fromarray(img.astype(np.uint8, copy=False), mode="RGB")
                bbox_mask = Image.fromarray(bbox_mask.astype(np.uint8, copy=False), mode="RGB")

            full_img = TF.resize(full_img, size=[224, 224])
            bbox_mask = TF.resize(bbox_mask, size=[224, 224])

            full_img = TF.to_tensor(full_img)
            resized_bbox_mask = TF.resize(bbox_mask, [32, 32])
            resized_bbox_mask = TF.to_tensor(resized_bbox_mask)[:2].float() / 255.0
            bbox_mask = TF.to_tensor(bbox_mask)[:2].float() / 255.0

            if self.norm_data:
                full_img = TF.normalize(full_img, mean=self.im_mean, std=self.im_std)

            datum["img"] = full_img
            datum['subject']['bbox'] = self.get_bbox_coord_from_mask(bbox_mask[0])
            datum['object']['bbox'] = self.get_bbox_coord_from_mask(bbox_mask[1])
            datum['predicate']['bbox'] = np.array(
                self._getUnionBBox(datum['subject']['bbox'], datum['object']['bbox'], ih=1, iw=1)
            )
            datum['bbox_mask'] = resized_bbox_mask

        return datum


def extract_subject_features(model, dataloader, device, desc="Extracting features"):
    """提取subject特征"""
    model.eval()
    features = []
    bboxes_s = []
    bboxes_o = []
    sample_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            rgb = batch['img'].to(device)
            subj_bbox = batch['subject']['bbox'].to(device)
            obj_bbox = batch['object']['bbox'].to(device)
            
            # 提取backbone特征
            image_feature = model.backbone(rgb)
            
            # 提取subject ROI特征
            img_h, img_w = rgb.shape[2], rgb.shape[3]
            rescaled_bbox_s = subj_bbox.clone()
            rescaled_bbox_s[:, 0] *= img_h
            rescaled_bbox_s[:, 1] *= img_h
            rescaled_bbox_s[:, 2] *= img_w
            rescaled_bbox_s[:, 3] *= img_w
            
            sub_feature = model.subject_feature_extractor(image_feature, rescaled_bbox_s)
            
            features.append(sub_feature.cpu())
            bboxes_s.append(subj_bbox.cpu())
            bboxes_o.append(obj_bbox.cpu())
            sample_ids.extend(batch['_id'])
    
    features = torch.cat(features, dim=0)
    bboxes_s = torch.cat(bboxes_s, dim=0)
    bboxes_o = torch.cat(bboxes_o, dim=0)
    return features, bboxes_s, bboxes_o, sample_ids


def compute_similarity_stats(features1, features2, name1, name2):
    """计算两组特征的相似度统计"""
    cosine_sims = []
    l2_dists = []
    
    for i in range(len(features1)):
        feat1 = features1[i]
        feat2 = features2[i]
        
        cos_sim = F.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0)).item()
        cosine_sims.append(cos_sim)
        
        l2_dist = torch.norm(feat1 - feat2, p=2).item()
        l2_dists.append(l2_dist)
    
    cosine_sims = np.array(cosine_sims)
    l2_dists = np.array(l2_dists)
    
    print(f"\n{name1} vs {name2}:")
    print(f"  Cosine Similarity: mean={cosine_sims.mean():.6f}, std={cosine_sims.std():.6f}, min={cosine_sims.min():.6f}, max={cosine_sims.max():.6f}")
    print(f"  L2 Distance: mean={l2_dists.mean():.6f}, std={l2_dists.std():.6f}, min={l2_dists.min():.6f}, max={l2_dists.max():.6f}")
    print(f"  Samples with Cosine Sim > 0.9999: {(cosine_sims > 0.9999).sum()}/{len(cosine_sims)} ({100*(cosine_sims > 0.9999).sum()/len(cosine_sims):.2f}%)")
    
    return cosine_sims, l2_dists


def verify_features_with_control(cfg, device, use_pretrained=True):
    """
    验证特征提取的确定性
    - 3次原图提取 (用于验证确定性)
    - 1次涂红提取 (实验组)
    """
    
    # 创建数据集
    print("Creating datasets...")
    dataset_original = SpatialSenseDatasetOriginal(
        split='test',
        predicate_dim=cfg.DATALOADER.predicate_dim,
        object_dim=cfg.DATALOADER.object_dim,
        data_path=cfg.DATALOADER.datapath,
        load_img=True,
        data_aug_shift=False,
        data_aug_color=False,
        crop=False,
        norm_data=True
    )
    
    dataset_red = SpatialSenseDatasetRedObject(
        split='test',
        predicate_dim=cfg.DATALOADER.predicate_dim,
        object_dim=cfg.DATALOADER.object_dim,
        data_path=cfg.DATALOADER.datapath,
        load_img=True,
        data_aug_shift=False,
        data_aug_color=False,
        crop=False,
        norm_data=True
    )
    
    # 创建数据加载器
    loader_original = DataLoader(dataset_original, batch_size=32, shuffle=False, num_workers=4)
    loader_red = DataLoader(dataset_red, batch_size=32, shuffle=False, num_workers=4)
    
    # 创建模型
    print(f"Building model (use_pretrained={use_pretrained})...")
    model = build_model(cfg)
    
    if not use_pretrained:
        print("⚠️  Using RANDOMLY initialized model (no pretrained weights)")
    else:
        print(f"✓ Using pretrained backbone from: {cfg.MODEL.REGIONVIT.pretrain_ckp}")
    
    model.to(device)
    model.eval()
    
    # 提取特征 - 3次原图
    print("\n" + "="*80)
    print("STEP 1: Extracting features from ORIGINAL images (3 times)")
    print("="*80)
    
    print("\n[Run 1/3] Original images...")
    features_orig_1, bboxes_s, bboxes_o, ids = extract_subject_features(model, loader_original, device, desc="Original Run 1")
    
    print("\n[Run 2/3] Original images...")
    features_orig_2, _, _, ids_2 = extract_subject_features(model, loader_original, device, desc="Original Run 2")
    
    print("\n[Run 3/3] Original images...")
    features_orig_3, _, _, ids_3 = extract_subject_features(model, loader_original, device, desc="Original Run 3")
    
    # 验证顺序一致
    assert ids == ids_2 == ids_3, "Sample IDs don't match across runs!"
    
    # 提取特征 - 1次涂红
    print("\n" + "="*80)
    print("STEP 2: Extracting features from RED OBJECT images (1 time)")
    print("="*80)
    
    print("\n[Red] Red object images...")
    features_red, _, _, ids_red = extract_subject_features(model, loader_red, device, desc="Red Object")
    
    assert ids == ids_red, "Sample IDs don't match between original and red!"
    
    # 计算IoU
    print("\n" + "="*80)
    print("STEP 3: Computing IoU and similarities")
    print("="*80)
    
    ious = []
    for i in range(len(ids)):
        iou = bbox_overlap(bboxes_s[i].numpy(), bboxes_o[i].numpy())
        ious.append(iou)
    ious = np.array(ious)
    
    # ========== 验证确定性 ==========
    print("\n" + "="*80)
    print("DETERMINISM CHECK: Comparing ORIGINAL runs with each other")
    print("="*80)
    print("If feature extraction is deterministic, these should be nearly identical (Cosine Sim ≈ 1.0)")
    
    cos_12, l2_12 = compute_similarity_stats(features_orig_1, features_orig_2, "Original Run 1", "Original Run 2")
    cos_13, l2_13 = compute_similarity_stats(features_orig_1, features_orig_3, "Original Run 1", "Original Run 3")
    cos_23, l2_23 = compute_similarity_stats(features_orig_2, features_orig_3, "Original Run 2", "Original Run 3")
    
    # ========== 实验组对比 ==========
    print("\n" + "="*80)
    print("EXPERIMENT: Comparing ORIGINAL vs RED OBJECT")
    print("="*80)
    print("This shows the effect of painting the object box red")
    
    cosine_sims, l2_dists = compute_similarity_stats(features_orig_1, features_red, "Original", "Red Object")
    
    # ========== 分组统计 ==========
    print("\n" + "="*80)
    print("STATISTICS BY IoU GROUPS")
    print("="*80)
    
    print(f"\nOverall IoU statistics:")
    print(f"  mean={ious.mean():.4f}, std={ious.std():.4f}, min={ious.min():.4f}, max={ious.max():.4f}")
    
    thresholds = [0.0, 0.1, 0.3, 0.5]
    for i in range(len(thresholds)):
        if i < len(thresholds) - 1:
            mask = (ious >= thresholds[i]) & (ious < thresholds[i+1])
            label = f"IoU in [{thresholds[i]:.1f}, {thresholds[i+1]:.1f})"
        else:
            mask = ious >= thresholds[i]
            label = f"IoU >= {thresholds[i]:.1f}"
        
        if mask.sum() > 0:
            print(f"\n{label} ({mask.sum()} samples):")
            print(f"  Original vs Red:")
            print(f"    Cosine Similarity: mean={cosine_sims[mask].mean():.6f}, std={cosine_sims[mask].std():.6f}")
            print(f"    L2 Distance: mean={l2_dists[mask].mean():.4f}, std={l2_dists[mask].std():.4f}")
    
    # ========== 关键发现 ==========
    no_overlap_mask = ious < 0.01
    print(f"\n{'*'*80}")
    print(f"CRITICAL: No Overlap Cases (IoU < 0.01): {no_overlap_mask.sum()} samples")
    print(f"{'*'*80}")
    
    if no_overlap_mask.sum() > 0:
        print(f"\n1. DETERMINISM (Original runs with each other):")
        print(f"   Run1 vs Run2: Cosine Sim = {cos_12[no_overlap_mask].mean():.6f}")
        print(f"   Run1 vs Run3: Cosine Sim = {cos_13[no_overlap_mask].mean():.6f}")
        print(f"   Run2 vs Run3: Cosine Sim = {cos_23[no_overlap_mask].mean():.6f}")
        
        print(f"\n2. EXPERIMENT (Original vs Red):")
        print(f"   Original vs Red: Cosine Sim = {cosine_sims[no_overlap_mask].mean():.6f}")
        print(f"   L2 Distance: {l2_dists[no_overlap_mask].mean():.4f}")
        
        print(f"\n3. COMPARISON:")
        avg_determinism = (cos_12[no_overlap_mask].mean() + cos_13[no_overlap_mask].mean() + cos_23[no_overlap_mask].mean()) / 3
        avg_experiment = cosine_sims[no_overlap_mask].mean()
        print(f"   Average similarity between original runs: {avg_determinism:.6f}")
        print(f"   Similarity between original and red: {avg_experiment:.6f}")
        print(f"   Difference: {avg_determinism - avg_experiment:.6f}")
    
    # ========== 可视化 ==========
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 第一行：确定性检查
    axes[0, 0].hist(cos_12, bins=50, edgecolor='black', alpha=0.7, color='blue')
    axes[0, 0].set_xlabel('Cosine Similarity')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Original Run 1 vs Run 2\n(Determinism Check)')
    axes[0, 0].axvline(x=0.9999, color='r', linestyle='--', label='0.9999')
    axes[0, 0].legend()
    
    axes[0, 1].hist(cos_13, bins=50, edgecolor='black', alpha=0.7, color='blue')
    axes[0, 1].set_xlabel('Cosine Similarity')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Original Run 1 vs Run 3\n(Determinism Check)')
    axes[0, 1].axvline(x=0.9999, color='r', linestyle='--', label='0.9999')
    axes[0, 1].legend()
    
    axes[0, 2].hist(cos_23, bins=50, edgecolor='black', alpha=0.7, color='blue')
    axes[0, 2].set_xlabel('Cosine Similarity')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('Original Run 2 vs Run 3\n(Determinism Check)')
    axes[0, 2].axvline(x=0.9999, color='r', linestyle='--', label='0.9999')
    axes[0, 2].legend()
    
    # 第二行：实验结果
    axes[1, 0].hist(cosine_sims, bins=50, edgecolor='black', alpha=0.7, color='red')
    axes[1, 0].set_xlabel('Cosine Similarity')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Original vs Red Object\n(Experiment)')
    axes[1, 0].axvline(x=cosine_sims.mean(), color='blue', linestyle='--', label=f'Mean={cosine_sims.mean():.3f}')
    axes[1, 0].legend()
    
    axes[1, 1].scatter(ious, cosine_sims, alpha=0.5, s=10, color='red')
    axes[1, 1].set_xlabel('IoU')
    axes[1, 1].set_ylabel('Cosine Similarity')
    axes[1, 1].set_title('Original vs Red: Similarity vs IoU')
    axes[1, 1].axvline(x=0.01, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3)
    
    # 对比不重叠样本
    no_overlap_orig = cos_12[ious < 0.01]
    no_overlap_exp = cosine_sims[ious < 0.01]
    
    axes[1, 2].hist([no_overlap_orig, no_overlap_exp], bins=30, 
                    label=['Original vs Original', 'Original vs Red'], 
                    alpha=0.7, edgecolor='black', color=['blue', 'red'])
    axes[1, 2].set_xlabel('Cosine Similarity')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title('No Overlap (IoU < 0.01) Comparison')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('feature_verification_with_control.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to: feature_verification_with_control.png")
    
    # ========== 保存详细数据 ==========
    results = {
        'ious': ious,
        'cosine_sims_orig_vs_red': cosine_sims,
        'l2_dists_orig_vs_red': l2_dists,
        'cosine_sims_orig1_vs_orig2': cos_12,
        'cosine_sims_orig1_vs_orig3': cos_13,
        'cosine_sims_orig2_vs_orig3': cos_23,
        'l2_dists_orig1_vs_orig2': l2_12,
        'l2_dists_orig1_vs_orig3': l2_13,
        'l2_dists_orig2_vs_orig3': l2_23,
        'sample_ids': ids
    }
    np.savez('feature_verification_with_control.npz', **results)
    print("Detailed data saved to: feature_verification_with_control.npz")
    
    # ========== 最终结论 ==========
    print("\n" + "="*80)
    print("FINAL CONCLUSION:")
    print("="*80)
    
    avg_determinism_all = (cos_12.mean() + cos_13.mean() + cos_23.mean()) / 3
    avg_experiment_all = cosine_sims.mean()
    
    print(f"\n1. DETERMINISM TEST:")
    print(f"   Average cosine similarity between original runs: {avg_determinism_all:.6f}")
    if avg_determinism_all > 0.9999:
        print("   ✓ Feature extraction is HIGHLY DETERMINISTIC")
    elif avg_determinism_all > 0.999:
        print("   ✓ Feature extraction is MOSTLY DETERMINISTIC (minor numerical variations)")
    else:
        print("   ✗ Feature extraction shows UNEXPECTED variations")
    
    print(f"\n2. EXPERIMENT RESULT:")
    print(f"   Cosine similarity between original and red: {avg_experiment_all:.6f}")
    print(f"   Difference from determinism baseline: {avg_determinism_all - avg_experiment_all:.6f}")
    
    if no_overlap_mask.sum() > 0:
        avg_det_no_overlap = (cos_12[no_overlap_mask].mean() + cos_13[no_overlap_mask].mean() + cos_23[no_overlap_mask].mean()) / 3
        avg_exp_no_overlap = cosine_sims[no_overlap_mask].mean()
        
        print(f"\n3. NO OVERLAP CASES (IoU < 0.01):")
        print(f"   Original runs similarity: {avg_det_no_overlap:.6f}")
        print(f"   Original vs Red similarity: {avg_exp_no_overlap:.6f}")
        print(f"   Effect of painting red: {avg_det_no_overlap - avg_exp_no_overlap:.6f} drop in similarity")
        
        if avg_det_no_overlap > 0.9999 and avg_exp_no_overlap < 0.95:
            print("\n   ✓ CONFIRMED: Painting the object box red DOES affect subject features,")
            print("     even when they don't overlap. This is NOT due to randomness.")
            print("     This suggests ViT's global attention mechanism is capturing the effect.")
        elif avg_det_no_overlap < 0.999:
            print("\n   ⚠ WARNING: Determinism check shows variations. Results may be affected by:")
            print("     - BatchNorm running statistics")
            print("     - Dropout (if not in eval mode)")
            print("     - GPU numerical precision")
        else:
            print("\n   ? UNCLEAR: Results require further investigation")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-config', type=str, required=True, help='Path to config file')
    parser.add_argument('--random-init', action='store_true', help='Use random initialization instead of pretrained backbone')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    
    args = parser.parse_args()
    
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.exp_config)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    print("Configuration:")
    print(cfg)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # 运行验证
    verify_features_with_control(cfg, device, use_pretrained=not args.random_init)

