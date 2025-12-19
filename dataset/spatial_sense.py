import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import json
import random
import numpy as np

import os.path
import cv2
import math
from PIL import Image, ImageDraw


IMAGE_SIZE = 224


class SpatialSenseDataset(Dataset):
    """
    修改版数据集：
    1. 将所有样本的subject移到图像中心
    2. 过滤subject和object重合的样本
    3. 将object区域涂红
    4. 用于验证模型是否依赖subject的绝对位置
    """
    def __init__(self, split, predicate_dim, object_dim, data_path, load_img,
                 data_aug_shift, data_aug_color, crop, norm_data):
        super().__init__()
        self.split = split
        self.load_image = load_img
        self.annotation_path = data_path
        self.img_path = os.path.join(os.path.dirname(data_path), 'images/')
        with open(self.annotation_path, 'r') as f:
            self.data = json.load(f)

        self.predicates = self.data['predicates']
        self.objects = self.data['objects']
        assert len(self.predicates) == predicate_dim  # 9
        assert len(self.objects) == object_dim  # 3679

        # 注意：禁用了shift augmentation，因为我们需要固定subject位置
        self.data_aug_shift = False  # 强制关闭
        self.data_aug_color = data_aug_color
        self.crop = crop
        self.norm_data = norm_data

        self.im_mean = [0.485, 0.456, 0.406]
        self.im_std = [0.229, 0.224, 0.225]

        # 统计信息
        self.total_samples = 0
        self.filtered_samples = 0  # 因重合被过滤的样本数
        self.overlap_threshold = 0.1  # IoU阈值，超过此值认为重合

        self.rgb_path_to_id = {}
        self.annotations = []
        self.annot_idx_each_predicate = {k:[] for k in self.predicates}
        
        idx = 0
        for img in self.data['sample_annots']:
            if img["split"] in split.split("_"):
                for annot in img["annotations"]:
                    self.total_samples += 1
                    annot["url"] = img["url"]
                    annot["height"] = img["height"]
                    annot["width"] = img["width"]
                    annot["subject"]["bbox"] = self.fix_bbox(
                        annot["subject"]["bbox"], img["height"], img["width"]
                    )
                    annot["object"]["bbox"] = self.fix_bbox(
                        annot["object"]["bbox"], img["height"], img["width"]
                    )
                    self.annotations.append(annot)
                    self.rgb_path_to_id[self.get_img_path(annot["url"], self.img_path)] = idx
                    self.annot_idx_each_predicate[annot['predicate']].append(idx)
                    idx += 1

        # Inpainting配置
        self.use_inpainting = True  # 默认启用inpainting
        
        # 保存inpainting图像的配置
        self.save_inpaint_samples = True  # 是否保存inpainting样本
        self.max_save_samples = 10  # 最多保存的样本数
        self.saved_samples_count = 0  # 已保存的样本计数
        self.save_dir = "./inpaint_samples"  # 保存目录
        if self.save_inpaint_samples:
            os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"{'='*70}")
        print(f"数据集初始化 - {split}")
        print(f"  总样本数: {self.total_samples}")
        print(f"  重合阈值: IoU > {self.overlap_threshold}")
        print(f"  Subject居中: 启用")
        print(f"  Object涂红: 启用")
        print(f"  Inpainting补全: {'启用' if self.use_inpainting else '禁用'}")
        if self.save_inpaint_samples:
            print(f"  保存inpainting样本: 启用 (最多{self.max_save_samples}张，保存到 {self.save_dir})")
        print(f"{'='*70}\n")

    def __len__(self):
        return len(self.annotations)

    def compute_iou(self, bbox1, bbox2):
        """
        计算两个bbox的IoU
        bbox格式: [y0, y1, x0, x1]
        """
        y0_1, y1_1, x0_1, x1_1 = bbox1
        y0_2, y1_2, x0_2, x1_2 = bbox2
        
        # 计算交集
        y0_inter = max(y0_1, y0_2)
        y1_inter = min(y1_1, y1_2)
        x0_inter = max(x0_1, x0_2)
        x1_inter = min(x1_1, x1_2)
        
        if y1_inter <= y0_inter or x1_inter <= x0_inter:
            return 0.0
        
        intersection = (y1_inter - y0_inter) * (x1_inter - x0_inter)
        
        # 计算并集
        area1 = (y1_1 - y0_1) * (x1_1 - x0_1)
        area2 = (y1_2 - y0_2) * (x1_2 - x0_2)
        union = area1 + area2 - intersection
        
        iou = intersection / (union + 1e-8)
        return iou

    def center_subject_and_translate(self, img, bbox_s, bbox_o, use_inpainting=True):
        """
        将subject移动到图像中心，并相应平移object
        
        Args:
            img: numpy array (H, W, 3)
            bbox_s: subject bbox [y0, y1, x0, x1]
            bbox_o: object bbox [y0, y1, x0, x1]
            use_inpainting: 是否使用inpainting补全缺失区域
            
        Returns:
            centered_img: 居中后的图像
            new_bbox_s: 平移后的subject bbox
            new_bbox_o: 平移后的object bbox
            is_valid: 是否有效（不重合）
        """
        ih, iw = img.shape[:2]
        
        # 1. 计算subject的中心点
        ys0, ys1, xs0, xs1 = bbox_s
        subject_center_y = (ys0 + ys1) / 2.0
        subject_center_x = (xs0 + xs1) / 2.0
        
        # 2. 计算需要的平移量（使subject中心对齐到图像中心）
        image_center_y = ih / 2.0
        image_center_x = iw / 2.0
        
        shift_y = image_center_y - subject_center_y
        shift_x = image_center_x - subject_center_x
        
        # 3. 应用平移到bbox
        new_bbox_s = [
            ys0 + shift_y,
            ys1 + shift_y,
            xs0 + shift_x,
            xs1 + shift_x
        ]
        
        yo0, yo1, xo0, xo1 = bbox_o
        new_bbox_o = [
            yo0 + shift_y,
            yo1 + shift_y,
            xo0 + shift_x,
            xo1 + shift_x
        ]
        
        # 4. 检查平移后是否重合
        iou = self.compute_iou(new_bbox_s, new_bbox_o)
        is_valid = (iou <= self.overlap_threshold)
        
        # 5. 应用平移到图像
        # 使用仿射变换矩阵
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        centered_img_before_inpaint = cv2.warpAffine(img, M, (iw, ih), 
                                                     borderMode=cv2.BORDER_CONSTANT, 
                                                     borderValue=(0, 0, 0))
        
        # 6. 使用inpainting补全黑色边界区域
        centered_img = centered_img_before_inpaint.copy()
        if use_inpainting:
            centered_img = self.inpaint_missing_regions(centered_img, new_bbox_s, new_bbox_o)
        
        # 7. 裁剪bbox到图像范围内
        new_bbox_s = [
            max(0, new_bbox_s[0]),
            min(ih, new_bbox_s[1]),
            max(0, new_bbox_s[2]),
            min(iw, new_bbox_s[3])
        ]
        
        new_bbox_o = [
            max(0, new_bbox_o[0]),
            min(ih, new_bbox_o[1]),
            max(0, new_bbox_o[2]),
            min(iw, new_bbox_o[3])
        ]
        
        # 8. 验证裁剪后的bbox是否有效（高度和宽度必须大于0）
        h_s = new_bbox_s[1] - new_bbox_s[0]
        w_s = new_bbox_s[3] - new_bbox_s[2]
        h_o = new_bbox_o[1] - new_bbox_o[0]
        w_o = new_bbox_o[3] - new_bbox_o[2]
        
        if h_s <= 0 or w_s <= 0 or h_o <= 0 or w_o <= 0:
            # 如果bbox无效，标记为无效样本
            is_valid = False
        
        return centered_img, new_bbox_s, new_bbox_o, is_valid, iou, centered_img_before_inpaint

    def inpaint_missing_regions(self, img, bbox_s, bbox_o):
        """
        使用OpenCV inpainting补全图像中的黑色边界区域
        
        Args:
            img: numpy array (H, W, 3)，可能包含黑色边界
            bbox_s: subject bbox [y0, y1, x0, x1]，需要保护的区域
            bbox_o: object bbox [y0, y1, x0, x1]，需要保护的区域
            
        Returns:
            inpainted_img: 补全后的图像
        """
        ih, iw = img.shape[:2]
        
        # 创建mask：标记需要inpaint的区域（黑色像素，但不包括subject和object区域）
        # 将图像转换为uint8格式用于处理
        img_uint8 = img.astype(np.uint8)
        
        # 检测黑色区域（RGB值接近(0,0,0)的区域）
        # 使用阈值来识别黑色像素
        black_threshold = 10
        mask = np.all(img_uint8 < black_threshold, axis=2).astype(np.uint8) * 255
        
        # 保护subject和object区域，不进行inpainting
        ys0, ys1, xs0, xs1 = [int(max(0, min(v, ih if i < 2 else iw))) 
                              for i, v in enumerate([bbox_s[0], bbox_s[1], bbox_s[2], bbox_s[3]])]
        yo0, yo1, xo0, xo1 = [int(max(0, min(v, ih if i < 2 else iw))) 
                              for i, v in enumerate([bbox_o[0], bbox_o[1], bbox_o[2], bbox_o[3]])]
        
        # 在mask中将subject和object区域设为0（不inpaint）
        if ys1 > ys0 and xs1 > xs0:
            mask[ys0:ys1, xs0:xs1] = 0
        if yo1 > yo0 and xo1 > xo0:
            mask[yo0:yo1, xo0:xo1] = 0
        
        # 如果mask中没有需要inpaint的区域，直接返回原图
        if np.sum(mask) == 0:
            return img
        
        # 使用OpenCV的inpainting算法
        # INPAINT_TELEA: 快速算法，适合大多数情况
        # INPAINT_NS: 更慢但质量更好
        inpainted_img = cv2.inpaint(img_uint8, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        
        # 转换回float32格式
        return inpainted_img.astype(np.float32)
    
    def paint_object_red(self, img, bbox_o):
        """
        将object区域涂成红色
        
        Args:
            img: numpy array (H, W, 3)
            bbox_o: object bbox [y0, y1, x0, x1]
            
        Returns:
            img: 涂红后的图像
        """
        yo0, yo1, xo0, xo1 = bbox_o
        yo0, yo1 = int(yo0), int(yo1)
        xo0, xo1 = int(xo0), int(xo1)
        
        # 确保bbox在图像范围内
        ih, iw = img.shape[:2]
        yo0 = max(0, min(yo0, ih))
        yo1 = max(0, min(yo1, ih))
        xo0 = max(0, min(xo0, iw))
        xo1 = max(0, min(xo1, iw))
        
        # 涂红
        if yo1 > yo0 and xo1 > xo0:
            img[yo0:yo1, xo0:xo1, :] = [255.0, 0.0, 0.0]  # RGB格式，红色
        
        return img
    
    def save_inpaint_sample(self, idx, original_img, before_inpaint_img, after_inpaint_img, 
                           bbox_s, bbox_o, annot):
        """
        保存inpainting样本的对比图像
        
        Args:
            idx: 样本索引
            original_img: 原始图像
            before_inpaint_img: inpainting前的图像（居中后）
            after_inpaint_img: inpainting后的图像
            bbox_s: subject bbox
            bbox_o: object bbox
            annot: 标注信息
        """
        try:
            # 将object区域涂红（用于最终图像）
            final_img = self.paint_object_red(after_inpaint_img.copy(), bbox_o)
            
            # 调整图像大小以便显示（统一缩放到224x224）
            target_size = 224
            original_resized = cv2.resize(original_img.astype(np.uint8), (target_size, target_size))
            before_inpaint_resized = cv2.resize(before_inpaint_img.astype(np.uint8), (target_size, target_size))
            after_inpaint_resized = cv2.resize(after_inpaint_img.astype(np.uint8), (target_size, target_size))
            final_resized = cv2.resize(final_img.astype(np.uint8), (target_size, target_size))
            
            # 创建对比图像（2x2布局）
            comparison = np.zeros((target_size * 2, target_size * 2, 3), dtype=np.uint8)
            comparison[0:target_size, 0:target_size] = original_resized
            comparison[0:target_size, target_size:target_size*2] = before_inpaint_resized
            comparison[target_size:target_size*2, 0:target_size] = after_inpaint_resized
            comparison[target_size:target_size*2, target_size:target_size*2] = final_resized
            
            # 添加文字标签
            comparison_pil = Image.fromarray(comparison)
            draw = ImageDraw.Draw(comparison_pil)
            
            # 添加标题
            font_size = 16
            try:
                from PIL import ImageFont
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                font = None
            
            labels = [
                ("原始图像", (10, 10)),
                ("居中后(inpaint前)", (target_size + 10, 10)),
                ("Inpainting后", (10, target_size + 10)),
                ("最终(涂红后)", (target_size + 10, target_size + 10))
            ]
            
            for label, pos in labels:
                draw.text(pos, label, fill=(255, 255, 0), font=font)
            
            # 添加bbox标注
            ih, iw = original_img.shape[:2]
            scale = target_size / max(ih, iw)
            
            # 在原始图像上画bbox
            ys0, ys1, xs0, xs1 = annot["subject"]["bbox"]
            yo0, yo1, xo0, xo1 = annot["object"]["bbox"]
            draw.rectangle(
                (xs0 * scale, ys0 * scale, xs1 * scale, ys1 * scale),
                outline='blue', width=2
            )
            draw.rectangle(
                (xo0 * scale, yo0 * scale, xo1 * scale, yo1 * scale),
                outline='red', width=2
            )
            
            # 在最终图像上画bbox（居中后的坐标）
            # 注意：bbox_s和bbox_o是居中后的坐标，需要按原始图像尺寸缩放
            ys0, ys1, xs0, xs1 = bbox_s
            yo0, yo1, xo0, xo1 = bbox_o
            final_scale = target_size / max(ih, iw)
            # 最终图像在右下角
            offset_x = target_size
            offset_y = target_size
            draw.rectangle(
                (offset_x + xs0 * final_scale, offset_y + ys0 * final_scale,
                 offset_x + xs1 * final_scale, offset_y + ys1 * final_scale),
                outline='blue', width=2
            )
            draw.rectangle(
                (offset_x + xo0 * final_scale, offset_y + yo0 * final_scale,
                 offset_x + xo1 * final_scale, offset_y + yo1 * final_scale),
                outline='red', width=2
            )
            
            # 在inpainting后的图像上也画bbox（左下角）
            draw.rectangle(
                (xs0 * final_scale, target_size + ys0 * final_scale,
                 xs1 * final_scale, target_size + ys1 * final_scale),
                outline='blue', width=2
            )
            draw.rectangle(
                (xo0 * final_scale, target_size + yo0 * final_scale,
                 xo1 * final_scale, target_size + yo1 * final_scale),
                outline='red', width=2
            )
            
            # 添加信息文本
            info_text = (
                f"Sample {idx}\n"
                f"Subject: {annot['subject']['name']}\n"
                f"Predicate: {annot['predicate']}\n"
                f"Object: {annot['object']['name']}"
            )
            draw.text((target_size + 10, target_size + 30), info_text, fill=(255, 255, 255), font=font)
            
            # 保存图像
            filename = os.path.join(self.save_dir, f"inpaint_sample_{idx:04d}.jpg")
            comparison_pil.save(filename)
            print(f"保存inpainting样本: {filename}")
            
        except Exception as e:
            print(f"保存inpainting样本失败 (idx={idx}): {e}")
    
    def __getitem__(self, idx, visualize=False):
        annot = self.annotations[idx]

        # ===== 读取图像 =====
        img = self.read_img(annot["url"], self.img_path)
        ih, iw = img.shape[:2]
        
        # ===== 应用subject居中和平移 =====
        centered_img, new_bbox_s, new_bbox_o, is_valid, iou, centered_img_before_inpaint = self.center_subject_and_translate(
            img, annot["subject"]["bbox"], annot["object"]["bbox"], 
            use_inpainting=self.use_inpainting
        )
        
        # ===== 如果重合，跳过此样本 =====
        if not is_valid:
            self.filtered_samples += 1
            # 返回None，需要在DataLoader中处理
            return None
        
        # ===== 保存inpainting样本（仅保存前几张） =====
        if self.save_inpaint_samples and self.saved_samples_count < self.max_save_samples:
            self.save_inpaint_sample(
                idx, img, centered_img_before_inpaint, centered_img, 
                new_bbox_s, new_bbox_o, annot
            )
            self.saved_samples_count += 1
        
        # ===== 将object区域涂红 =====
        centered_img_after_red = self.paint_object_red(centered_img.copy(), new_bbox_o)
        
        # ===== 使用居中后的bbox =====
        bbox_s = new_bbox_s
        bbox_o = new_bbox_o

        # ===== 计算其他特征 =====
        t_s = self._getT(bbox_s, bbox_o)
        t_o = self._getT(bbox_o, bbox_s)

        ys0, ys1, xs0, xs1 = bbox_s
        yo0, yo1, xo0, xo1 = bbox_o
        union_bbox = self._getUnionBBox(bbox_s, bbox_o, ih, iw)

        datum = {
            "url": annot["url"],
            "_id": annot["_id"],
            "subject": {
                "name": annot["subject"]["name"],
                "idx": self.objects.index(annot['subject']['name']),
                "bbox": np.asarray(
                    [
                        ys0 / ih,
                        ys1 / ih,
                        xs0 / iw,
                        xs1 / iw,
                    ],
                    dtype=np.float32,
                ),
                "t": np.asarray(t_s, dtype=np.float32),
            },
            "object": {
                "name": annot["object"]["name"],
                "idx": self.objects.index(annot['object']['name']),
                "bbox": np.asarray(
                    [
                        yo0 / ih,
                        yo1 / ih,
                        xo0 / iw,
                        xo1 / iw,
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
                        union_bbox[0] / ih,
                        union_bbox[1] / ih,
                        union_bbox[2] / iw,
                        union_bbox[3] / iw,
                    ], dtype=np.float32,
                )
            },
            'rgb_source': self.get_img_path(annot["url"], self.img_path),
            'iou': iou,  # 记录IoU用于分析
        }

        if self.load_image:
            # 使用居中后的图像（涂红后）
            img = centered_img_after_red
            
            bbox_mask = np.stack(
                [
                    self._getDualMask(ih, iw, bbox_s, ih, iw).astype(np.uint8),
                    self._getDualMask(ih, iw, bbox_o, ih, iw).astype(np.uint8),
                    np.zeros((ih, iw), dtype=np.uint8),
                ],
                axis=2,
            )

            if self.crop:
                enlarged_union_bbox = self.enlarge(union_bbox, 1.25, ih, iw)
                full_img = Image.fromarray(img[enlarged_union_bbox[0]:enlarged_union_bbox[1], 
                                              enlarged_union_bbox[2]:enlarged_union_bbox[3], :].astype(np.uint8, copy=False), mode="RGB")
                bbox_mask = Image.fromarray(bbox_mask[enlarged_union_bbox[0]:enlarged_union_bbox[1], 
                                                     enlarged_union_bbox[2]:enlarged_union_bbox[3], :].astype(np.uint8, copy=False), mode="RGB")
            else:
                full_img = Image.fromarray(img.astype(np.uint8, copy=False), mode="RGB")
                bbox_mask = Image.fromarray(bbox_mask.astype(np.uint8, copy=False), mode="RGB")

            # 数据增强（注意：已禁用shift augmentation）
            if "train" in self.split:
                # 不进行shift augmentation，保持subject居中
                full_img = TF.resize(full_img, size=[IMAGE_SIZE, IMAGE_SIZE])
                bbox_mask = TF.resize(bbox_mask, size=[IMAGE_SIZE, IMAGE_SIZE])
                
                if self.data_aug_color:
                    full_img = TF.adjust_brightness(full_img, random.uniform(0.9, 1.1))
                    full_img = TF.adjust_contrast(full_img, random.uniform(0.9, 1.1))
                    full_img = TF.adjust_gamma(full_img, random.uniform(0.9, 1.1))
                    full_img = TF.adjust_hue(full_img, random.uniform(-0.05, 0.05))
            else:
                full_img = TF.resize(full_img, size=[IMAGE_SIZE, IMAGE_SIZE])
                bbox_mask = TF.resize(bbox_mask, size=[IMAGE_SIZE, IMAGE_SIZE])

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

            if visualize:
                self._visualize_sample(idx, img, full_img, bbox_s, bbox_o, annot, iou)

        return datum

    def _visualize_sample(self, idx, raw_img, processed_img, bbox_s, bbox_o, annot, iou):
        """可视化样本"""
        vis = Image.new(mode='RGB', size=(224 * 3, 224))
        
        # 原始图像
        raw_img_pil = Image.fromarray(raw_img.astype(np.uint8))
        raw_img_pil = raw_img_pil.resize(size=(224, 224))
        
        # 处理后的图像
        input_img = ((processed_img * torch.tensor(self.im_std).view(-1, 1, 1) + 
                     torch.tensor(self.im_mean).view(-1, 1, 1)) * 255).permute((1, 2, 0)).numpy().astype(np.uint8)
        input_img_obj = Image.fromarray(input_img)
        
        # 画bbox
        draw = ImageDraw.Draw(input_img_obj)
        ih, iw = raw_img.shape[:2]
        ys0, ys1, xs0, xs1 = bbox_s
        yo0, yo1, xo0, xo1 = bbox_o
        draw.rectangle(((xs0 * 224 / iw, ys0 * 224 / ih), (xs1 * 224 / iw, ys1 * 224 / ih)), 
                      outline='blue', width=2)
        draw.rectangle(((xo0 * 224 / iw, yo0 * 224 / ih), (xo1 * 224 / iw, yo1 * 224 / ih)), 
                      outline='red', width=2)
        
        vis.paste(raw_img_pil, (0, 0))
        vis.paste(input_img_obj, (224, 0))
        
        # 添加文字信息
        draw = ImageDraw.Draw(vis)
        draw.text(
            (224 * 2, 0),
            f"{annot['subject']['name']}\n"
            f"--{annot['predicate']}--\n"
            f"{annot['object']['name']}\n"
            f"Label: {annot['label']}\n"
            f"IoU: {iou:.3f}",
            (255, 255, 255),
        )
        
        vis.save(f'vis_centered_{idx:04d}.jpg')
        print(f"保存可视化: vis_centered_{idx:04d}.jpg")

    @staticmethod
    def get_img_path(url, imagepath):
        if url.startswith("http"):  # flickr
            filename = os.path.join(imagepath, "flickr", url.split("/")[-1])
        else:  # nyu
            filename = os.path.join(imagepath, "nyu", url.split("/")[-1])
        return filename

    def read_img(self, url, imagepath):
        filename = self.get_img_path(url, imagepath)
        img = cv2.imread(filename).astype(np.float32, copy=False)[:, :, ::-1]
        assert img.shape[2] == 3
        return img

    def enlarge(self, bbox, factor, ih, iw):
        height = bbox[1] - bbox[0]
        width = bbox[3] - bbox[2]
        assert height > 0 and width > 0
        return [
            max(0, int(bbox[0] - (factor - 1.0) * height / 2.0)),
            min(ih, int(bbox[1] + (factor - 1.0) * height / 2.0)),
            max(0, int(bbox[2] - (factor - 1.0) * width / 2.0)),
            min(iw, int(bbox[3] + (factor - 1.0) * width / 2.0)),
        ]

    @staticmethod
    def _getUnionBBox(aBB, bBB, ih, iw, margin=10):
        return [
            max(0, min(aBB[0], bBB[0]) - margin),
            min(ih, max(aBB[1], bBB[1]) + margin),
            max(0, min(aBB[2], bBB[2]) - margin),
            min(iw, max(aBB[3], bBB[3]) + margin),
        ]

    @staticmethod
    def _getDualMask(ih, iw, bb, heatmap_h=32, heatmap_w=32):
        rh = float(heatmap_h) / ih
        rw = float(heatmap_w) / iw
        x1 = max(0, int(math.floor(bb[0] * rh)))
        x2 = min(heatmap_h, int(math.ceil(bb[1] * rh)))
        y1 = max(0, int(math.floor(bb[2] * rw)))
        y2 = min(heatmap_w, int(math.ceil(bb[3] * rw)))
        mask = np.zeros((heatmap_h, heatmap_w), dtype=np.float32)
        mask[x1:x2, y1:y2] = 255
        return mask

    @staticmethod
    def get_bbox_coord_from_mask(mask):
        assert mask.dim() == 2
        h, w = mask.shape
        nz_mask = mask.nonzero()
        if len(nz_mask) != 0:
            t, l = nz_mask.min(0).values
            b, r = nz_mask.max(0).values
        else:
            t, l, b, r = [0] * 4

        return np.array([
            float(t) / h, float(b) / h,
            float(l) / w, float(r) / w
        ], np.float32)

    @staticmethod
    def _getT(bbox1, bbox2):
        h1 = bbox1[1] - bbox1[0]
        w1 = bbox1[3] - bbox1[2]
        h2 = bbox2[1] - bbox2[0]
        w2 = bbox2[3] - bbox2[2]
        
        # 安全检查：确保bbox尺寸为正数
        if h1 <= 0 or w1 <= 0 or h2 <= 0 or w2 <= 0:
            # 如果bbox无效，返回默认值
            return [0.0, 0.0, 0.0, 0.0]
        
        # 计算比值，确保为正数
        h_ratio = h1 / float(h2)
        w_ratio = w1 / float(w2)
        
        # 确保比值为正数（虽然理论上应该总是正数，但添加检查以防万一）
        if h_ratio <= 0 or w_ratio <= 0:
            return [0.0, 0.0, 0.0, 0.0]
        
        return [
            (bbox1[0] - bbox2[0]) / float(h2),
            (bbox1[2] - bbox2[2]) / float(w2),
            math.log(h_ratio),
            math.log(w_ratio),
        ]

    @staticmethod
    def fix_bbox(bbox, ih, iw):
        if bbox[1] - bbox[0] < 20:
            if bbox[0] > 10:
                bbox[0] -= 10
            if bbox[1] < ih - 10:
                bbox[1] += 10

        if bbox[3] - bbox[2] < 20:
            if bbox[2] > 10:
                bbox[2] -= 10
            if bbox[3] < iw - 10:
                bbox[3] += 10
        return bbox


def collate_fn_filter_none(batch):
    """
    自定义collate函数，过滤掉None样本
    """
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

