import json
import sys
sys.path.append('..')

from utils.bbox_utils import compute_iou, print_filter_stats
import numpy as np

# 数据路径
data_path = "./data/spatialsense/annots_spatialsenseplus.json"

print("\n分析 IoU 分布...\n")

# 加载数据
with open(data_path, 'r') as f:
    data = json.load(f)

print(f"数据集包含 {len(data['sample_annots'])} 张图片\n")

# 分析每个split
for split in ['train', 'valid', 'test']:
    print(f"\n{'='*70}")
    print(f"Split: {split.upper()}")
    print(f"{'='*70}")
    
    # 收集所有标注
    all_annots = []
    iou_list = []
    
    for img in data['sample_annots']:
        if img['split'] == split:
            for annot in img['annotations']:
                all_annots.append(annot)
                iou = compute_iou(
                    annot['subject']['bbox'],
                    annot['object']['bbox']
                )
                iou_list.append(iou)
    
    if len(all_annots) == 0:
        print(f"⚠️  没有找到 split='{split}' 的数据\n")
        continue
    
    # 统计
    iou_array = np.array(iou_list)
    filtered_count = (iou_array == 0).sum()
    
    stats = {
        'total': len(all_annots),
        'filtered': int(filtered_count),
        'ratio': float(filtered_count) / len(all_annots),
        'iou_mean': float(iou_array.mean()),
        'iou_distribution': {
            'iou_0.0': int((iou_array == 0).sum()),
            'iou_0.0-0.1': int(((iou_array > 0) & (iou_array <= 0.1)).sum()),
            'iou_0.1-0.3': int(((iou_array > 0.1) & (iou_array <= 0.3)).sum()),
            'iou_0.3-1.0': int((iou_array > 0.3).sum()),
        }
    }
    
    print_filter_stats(stats, split)
    
    # 按关系统计IoU=0的样本
    print(f"各关系的 IoU=0 样本数:")
    print("-"*50)
    relation_counts = {}
    for annot, iou in zip(all_annots, iou_list):
        if iou == 0:
            rel = annot['predicate']
            relation_counts[rel] = relation_counts.get(rel, 0) + 1
    
    for rel, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {rel:20s}: {count:5d}")
    print()

print("✅ 分析完成\n")
