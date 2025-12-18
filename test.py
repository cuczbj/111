import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

class FixedBBoxVisualizer:
    """
    为所有9种关系生成可视化
    验证固定bbox + 变化内容的设计
    """
    
    def __init__(self, img_size=224):
        self.img_size = img_size
        self.predicates = [
            "above", "behind", "in", "in front of", 
            "next to", "on", "to the left of", 
            "to the right of", "under"
        ]
        
        # ===== 固定的bbox（所有关系都用这些！）=====
        # 归一化坐标 [y0, y1, x0, x1]
        self.fixed_subject_bbox = np.array([0.25, 0.55, 0.25, 0.75])
        self.fixed_object_bbox = np.array([0.45, 0.75, 0.25, 0.75])
        self.fixed_union_bbox = np.array([0.25, 0.75, 0.25, 0.75])
        
        # 转换为像素坐标
        self.subject_bbox_px = (self.fixed_subject_bbox * img_size).astype(int)
        self.object_bbox_px = (self.fixed_object_bbox * img_size).astype(int)
        self.union_bbox_px = (self.fixed_union_bbox * img_size).astype(int)
        
        print("固定的BBox坐标：")
        print(f"  Subject: {self.fixed_subject_bbox} → 像素 {self.subject_bbox_px}")
        print(f"  Object:  {self.fixed_object_bbox} → 像素 {self.object_bbox_px}")
        print(f"  Union:   {self.fixed_union_bbox} → 像素 {self.union_bbox_px}\n")
    
    def generate_image_for_relation(self, relation):
        """
        为特定关系生成图像
        所有关系的bbox完全相同，只改变视觉内容
        """
        img = Image.new('RGB', (self.img_size, self.img_size), color='white')
        draw = ImageDraw.Draw(img)
        
        # Union区域的中心和尺寸
        y0u, y1u, x0u, x1u = self.union_bbox_px
        union_center_x = (x0u + x1u) // 2
        union_center_y = (y0u + y1u) // 2
        union_height = y1u - y0u
        union_width = x1u - x0u
        
        # 基础物体大小
        obj_size = 30
        
        # ===== 根据关系设计视觉内容 =====
        
        if relation == "above":
            # 红圆在上1/4，蓝方在下1/4
            subject_pos = (union_center_x, y0u + union_height // 4)
            object_pos = (union_center_x, y0u + 3 * union_height // 4)
            self.draw_circle(draw, subject_pos, obj_size, 'red')
            self.draw_square(draw, object_pos, obj_size, 'blue')
        
        elif relation == "under":
            # 红圆在下，蓝方在上（与above相反）
            subject_pos = (union_center_x, y0u + 3 * union_height // 4)
            object_pos = (union_center_x, y0u + union_height // 4)
            self.draw_circle(draw, subject_pos, obj_size, 'red')
            self.draw_square(draw, object_pos, obj_size, 'blue')
        
        elif relation == "to the left of":
            # 红圆在左，蓝方在右
            subject_pos = (x0u + union_width // 4, union_center_y)
            object_pos = (x0u + 3 * union_width // 4, union_center_y)
            self.draw_circle(draw, subject_pos, obj_size, 'red')
            self.draw_square(draw, object_pos, obj_size, 'blue')
        
        elif relation == "to the right of":
            # 红圆在右，蓝方在左
            subject_pos = (x0u + 3 * union_width // 4, union_center_y)
            object_pos = (x0u + union_width // 4, union_center_y)
            self.draw_circle(draw, subject_pos, obj_size, 'red')
            self.draw_square(draw, object_pos, obj_size, 'blue')
        
        elif relation == "on":
            # 红圆在蓝方正上方，接触但不遮挡
            object_pos = (union_center_x, y0u + 3 * union_height // 4)
            subject_pos = (union_center_x, object_pos[1] - obj_size)  # 刚好接触
            self.draw_square(draw, object_pos, obj_size, 'blue')  # 先画蓝方
            self.draw_circle(draw, subject_pos, obj_size, 'red')   # 再画红圆
        
        elif relation == "in front of":
            # 通过大小和遮挡表示：红圆更大，部分遮挡蓝方
            object_pos = (union_center_x + 5, union_center_y + 5)
            subject_pos = (union_center_x - 5, union_center_y - 5)
            # 先画object（在后面）
            self.draw_square(draw, object_pos, obj_size - 5, 'blue')
            # 再画subject（在前面，更大）
            self.draw_circle(draw, subject_pos, obj_size + 5, 'red')
        
        elif relation == "behind":
            # 红圆更小，被蓝方部分遮挡
            subject_pos = (union_center_x + 5, union_center_y + 5)
            object_pos = (union_center_x - 5, union_center_y - 5)
            # 先画subject（在后面，更小）
            self.draw_circle(draw, subject_pos, obj_size - 5, 'red')
            # 再画object（在前面，更大）
            self.draw_square(draw, object_pos, obj_size + 5, 'blue')
        
        elif relation == "in":
            # 红圆在蓝方内部
            object_pos = (union_center_x, union_center_y)
            subject_pos = (union_center_x, union_center_y)
            # 先画大的蓝方
            self.draw_square(draw, object_pos, obj_size + 15, 'blue')
            # 再画小的红圆（在内部）
            self.draw_circle(draw, subject_pos, obj_size - 10, 'red')
        
        elif relation == "next to":
            # 红圆和蓝方并排（水平相邻）
            subject_pos = (union_center_x - obj_size // 2 - 2, union_center_y)
            object_pos = (union_center_x + obj_size // 2 + 2, union_center_y)
            self.draw_circle(draw, subject_pos, obj_size, 'red')
            self.draw_square(draw, object_pos, obj_size, 'blue')
        
        return img
    
    def draw_circle(self, draw, center, size, color):
        """画圆"""
        x, y = center
        draw.ellipse([
            x - size//2, y - size//2,
            x + size//2, y + size//2
        ], fill=color, outline=color)
    
    def draw_square(self, draw, center, size, color):
        """画方块"""
        x, y = center
        draw.rectangle([
            x - size//2, y - size//2,
            x + size//2, y + size//2
        ], fill=color, outline=color)
    
    def draw_bbox_on_image(self, ax, img):
        """在图像上画bbox"""
        ax.imshow(img)
        
        # Subject bbox (红色边框)
        y0, y1, x0, x1 = self.subject_bbox_px
        ax.add_patch(Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=False, edgecolor='red', linewidth=2, 
            linestyle='--', label='Subject BBox'
        ))
        
        # Object bbox (蓝色边框)
        y0, y1, x0, x1 = self.object_bbox_px
        ax.add_patch(Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=False, edgecolor='blue', linewidth=2,
            linestyle='--', label='Object BBox'
        ))
        
        # Union bbox (绿色边框)
        y0, y1, x0, x1 = self.union_bbox_px
        ax.add_patch(Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=False, edgecolor='green', linewidth=3,
            linestyle='-', label='Union BBox'
        ))
    
    def visualize_all_relations(self, save_path='./fixed_bbox_visualization'):
        """生成所有关系的可视化"""
        os.makedirs(save_path, exist_ok=True)
        
        # 创建3x3的图像网格
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        print("="*70)
        print("正在生成可视化...")
        print("="*70)
        
        for idx, relation in enumerate(self.predicates):
            print(f"\n{idx+1}. 生成 '{relation}' 的图像...")
            
            # 生成图像
            img = self.generate_image_for_relation(relation)
            
            # 画bbox
            ax = axes[idx]
            self.draw_bbox_on_image(ax, img)
            ax.set_title(f'{relation}\n(BBox位置完全相同)', fontsize=12, fontweight='bold')
            ax.axis('off')
            
            # 只在第一个子图显示图例
            if idx == 0:
                ax.legend(loc='upper left', fontsize=8)
            
            # 单独保存每个关系的图像
            single_fig, single_ax = plt.subplots(1, 1, figsize=(6, 6))
            self.draw_bbox_on_image(single_ax, img)
            single_ax.set_title(f'{relation}', fontsize=14, fontweight='bold')
            single_ax.legend(loc='upper left')
            single_ax.axis('off')
            plt.tight_layout()
            plt.savefig(f'{save_path}/relation_{idx+1}_{relation.replace(" ", "_")}.png', 
                       dpi=150, bbox_inches='tight')
            plt.close(single_fig)
            
            print(f"   ✅ 已保存: {save_path}/relation_{idx+1}_{relation.replace(' ', '_')}.png")
        
        # 保存总览图
        plt.tight_layout()
        overview_path = f'{save_path}/00_all_relations_overview.png'
        plt.savefig(overview_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ 总览图已保存: {overview_path}")
        plt.show()
        
        print("\n" + "="*70)
        print("可视化完成！")
        print("="*70)
        print("\n📂 文件位置：")
        print(f"   {save_path}/")
        print("\n📋 关键验证点：")
        print("   1. 所有图像的3个bbox位置是否完全相同？")
        print("   2. 不同关系是否通过视觉内容（而非bbox位置）区分？")
        print("   3. Subject(红圆)和Object(蓝方)是否都在对应的bbox内？")
        print("\n如果以上都满足，说明设计正确！✅")
    
    def print_design_summary(self):
        """打印设计说明"""
        print("\n" + "="*70)
        print("🎨 设计说明")
        print("="*70)
        print("\n各关系的视觉表示：")
        print("  1. above:          红圆在上，蓝方在下（垂直排列）")
        print("  2. under:          红圆在下，蓝方在上（above的反向）")
        print("  3. to the left of: 红圆在左，蓝方在右（水平排列）")
        print("  4. to the right of:红圆在右，蓝方在左（left的反向）")
        print("  5. on:             红圆在蓝方正上方接触")
        print("  6. in front of:    红圆更大，部分遮挡蓝方（深度）")
        print("  7. behind:         红圆更小，被蓝方遮挡（深度）")
        print("  8. in:             红圆小，在蓝方内部（包含）")
        print("  9. next to:        红圆和蓝方水平并排")
        
        print("\n🔑 关键设计：")
        print("  ✅ Subject BBox: [0.25, 0.55, 0.25, 0.75] - 所有关系相同")
        print("  ✅ Object BBox:  [0.45, 0.75, 0.25, 0.75] - 所有关系相同")
        print("  ✅ Union BBox:   [0.25, 0.75, 0.25, 0.75] - 所有关系相同")
        print("\n  ⚠️  只有框内的视觉内容不同！")
        print("="*70 + "\n")


# ===== 主程序 =====
if __name__ == '__main__':
    print("🚀 固定BBox可视化工具")
    print("="*70)
    
    # 创建可视化器
    visualizer = FixedBBoxVisualizer(img_size=224)
    
    # 打印设计说明
    visualizer.print_design_summary()
    
    # 生成所有关系的可视化
    visualizer.visualize_all_relations(save_path='./fixed_bbox_visualization')
    
    print("\n🎯 下一步：")
    print("   1. 查看生成的图像，验证设计是否合理")
    print("   2. 确认bbox位置是否完全固定")
    print("   3. 检查不同关系是否能通过视觉区分")
    print("   4. 如果有问题，告诉我需要如何调整！")
