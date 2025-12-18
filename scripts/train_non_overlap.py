import os

pretrained_vit_path = "./data/pretrained_checkpoints/ibot_vit_base_patch16.pth"
data_path = "./data/spatialsense/annots_spatialsenseplus.json"
config_path = "configs/spatialsense/regionvit.yaml"  # 使用原有配置

print("\n" + "="*70)
print("🧪 实验：只使用不重合样本 (IoU=0)")
print("="*70)
print("\n对比实验：")
print("  1. Baseline: Subject-Only 无标记")
print("  2. 实验组: Subject-Only + Object标记")
print("\n" + "="*70 + "\n")

# 对比实验
experiments = [
    {
        'name': 'NonOverlap_NoMark',
        'mark_object': False,
        'desc': 'Subject-Only 无标记',
    },
    {
        'name': 'NonOverlap_WithMark',
        'mark_object': True,
        'desc': 'Subject-Only + Object标记',
    },
]

for exp in experiments:
    print(f"\n{'='*70}")
    print(f"🚀 实验: {exp['name']}")
    print(f"   {exp['desc']}")
    print(f"{'='*70}\n")
    
    for seed in range(5):
        exp_id = f"{exp['name']}_seed{seed}"
        
        command = f"""CUDA_VISIBLE_DEVICES=0 python main.py \
            --exp-config {config_path} \
            EXP.SEED {seed} \
            EXP.MODEL_NAME regionvit \
            EXP.EXP_ID {exp_id} \
            DATALOADER.datapath {data_path} \
            DATALOADER.filter_overlap True \
            DATALOADER.iou_threshold 0.0 \
            DATALOADER.mark_object {exp['mark_object']} \
            DATALOADER.mark_alpha 0.0 \
            MODEL.REGIONVIT.pretrain_ckp {pretrained_vit_path}"""
        
        print(f"Running seed {seed}...")
        ret = os.system(command)
        
        if ret != 0:
            print(f"\n❌ Error at seed {seed}")
            exit(1)
    
    print(f"\n✅ 完成: {exp['name']}\n")

print("\n" + "="*70)
print("🎉 所有实验完成！")
print("="*70 + "\n")
