# """
# SP-SAM Multimask + Prototype 组合版本
# =====================================

# 组合两个有效的改进：
# 1. multimask_output=True (60.33%)
# 2. Prototype相似度计算 (59.99%)

# 目标: 60%+ mIoU
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import cv2
# from PIL import Image
# from typing import List, Dict, Tuple, Optional, Any
# from tqdm import tqdm


# class CMRSModule:
#     """
#     CMRS模块 - Multimask + Prototype组合
#     """
    
#     def __init__(self, dino_model, dino_transform, device='cuda'):
#         self.dino_model = dino_model
#         self.dino_transform = dino_transform
#         self.device = device
        
#     def extract_dino_features(self, img_pil: Image.Image) -> torch.Tensor:
#         img_tensor = self.dino_transform(img_pil)[None].to(self.device)
        
#         with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
#             features = self.dino_model.get_intermediate_layers(
#                 img_tensor.to(torch.bfloat16)
#             )[0]
            
#             h = w = int(features.shape[1]**0.5)
#             feature_map = features.reshape(1, h, w, -1).permute(0, 3, 1, 2)
#             feature_map = feature_map.float()
            
#         return feature_map.squeeze(0)
    
#     def downsample_mask(self, mask_np: np.ndarray, target_size: Tuple[int, int]) -> torch.Tensor:
#         h, w = target_size
#         resized_mask = cv2.resize(
#             mask_np.astype(np.float32),
#             (w, h),
#             interpolation=cv2.INTER_NEAREST
#         )
#         resized_mask = (resized_mask > 0.5).astype(np.float32)
#         return torch.from_numpy(resized_mask).to(self.device)
    
#     def compute_prototype(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#         """Masked Average Pooling"""
#         mask_sum = mask.sum() + 1e-8
#         prototype = (features * mask.unsqueeze(0)).sum(dim=(1, 2)) / mask_sum
#         return prototype
    
#     def compute_similarity_map_prototype(self, query_features: torch.Tensor, 
#                                          support_features_list: List[torch.Tensor],
#                                          support_masks_list: List[torch.Tensor]) -> torch.Tensor:
#         """纯Prototype方法"""
#         C, H, W = query_features.shape
        
#         # 计算每个support的prototype
#         prototypes = []
#         for sup_feat, sup_mask in zip(support_features_list, support_masks_list):
#             proto = self.compute_prototype(sup_feat, sup_mask)
#             prototypes.append(proto)
        
#         # 合并prototypes
#         prototypes = torch.stack(prototypes, dim=0)
#         avg_prototype = prototypes.mean(dim=0)
#         avg_prototype = F.normalize(avg_prototype, p=2, dim=0)
        
#         # 计算相似度
#         query_flat = query_features.reshape(C, -1).T
#         query_flat = F.normalize(query_flat, p=2, dim=1)
        
#         similarity_scores = torch.mv(query_flat, avg_prototype)
#         similarity_map = similarity_scores.reshape(H, W)
#         return similarity_map
    
#     def get_prompts_from_similarity(self, similarity_map: torch.Tensor, 
#                                     top_k: int = 10,
#                                     neg_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
#         sim_np = similarity_map.cpu().numpy()
#         H, W = sim_np.shape
        
#         mean_sim = sim_np.mean()
#         std_sim = sim_np.std()
#         max_sim = sim_np.max()
        
#         fg_threshold = max(mean_sim + 1.0 * std_sim, max_sim * 0.5)
#         fg_mask = sim_np > fg_threshold
#         fg_coords = np.argwhere(fg_mask)
        
#         if len(fg_coords) < top_k:
#             fg_threshold = max(mean_sim + 0.5 * std_sim, max_sim * 0.4)
#             fg_mask = sim_np > fg_threshold
#             fg_coords = np.argwhere(fg_mask)
        
#         if len(fg_coords) < top_k:
#             fg_threshold = mean_sim + 0.3 * std_sim
#             fg_mask = sim_np > fg_threshold
#             fg_coords = np.argwhere(fg_mask)
        
#         if len(fg_coords) == 0:
#             flat_indices = np.argsort(sim_np.flatten())[-top_k:]
#             pos_points = np.array([np.unravel_index(idx, sim_np.shape) 
#                                   for idx in flat_indices])
#         else:
#             pos_points = self._sample_points_weighted_spatial(fg_coords, sim_np, top_k)
        
#         bg_threshold = mean_sim - 0.5 * std_sim
#         bg_mask = sim_np < bg_threshold
#         bg_coords = np.argwhere(bg_mask)
        
#         if len(bg_coords) >= neg_k:
#             neg_points = self._sample_negative_points_diverse(bg_coords, sim_np, neg_k, (H, W))
#         else:
#             neg_points = self._sample_from_borders(H, W, neg_k)
        
#         all_points = np.vstack([pos_points, neg_points])
#         pos_labels = np.ones(len(pos_points), dtype=np.int32)
#         neg_labels = np.zeros(len(neg_points), dtype=np.int32)
#         all_labels = np.concatenate([pos_labels, neg_labels])
        
#         point_coords = all_points[:, ::-1]
        
#         return point_coords, all_labels
    
#     def _sample_points_weighted_spatial(self, coords, sim_np, num_points):
#         sim_values = np.array([sim_np[y, x] for y, x in coords])
        
#         selected = []
#         selected_indices = []
        
#         first_idx = np.argmax(sim_values)
#         selected.append(coords[first_idx])
#         selected_indices.append(first_idx)
        
#         for _ in range(num_points - 1):
#             if len(selected_indices) >= len(coords):
#                 break
                
#             best_score = -np.inf
#             best_idx = -1
            
#             for i, (coord, sim_val) in enumerate(zip(coords, sim_values)):
#                 if i in selected_indices:
#                     continue
                
#                 min_dist = np.inf
#                 for sel_coord in selected:
#                     dist = np.sqrt((coord[0] - sel_coord[0])**2 + 
#                                   (coord[1] - sel_coord[1])**2)
#                     min_dist = min(min_dist, dist)
                
#                 score = sim_val * 0.7 + (min_dist / max(sim_np.shape)) * 0.3
                
#                 if score > best_score:
#                     best_score = score
#                     best_idx = i
            
#             if best_idx >= 0:
#                 selected.append(coords[best_idx])
#                 selected_indices.append(best_idx)
        
#         return np.array(selected)
    
#     def _sample_negative_points_diverse(self, bg_coords, sim_np, num_points, shape):
#         H, W = shape
#         grid_h, grid_w = 3, 3
#         cell_h, cell_w = H // grid_h, W // grid_w
        
#         neg_points = []
        
#         for gh in range(grid_h):
#             for gw in range(grid_w):
#                 if len(neg_points) >= num_points:
#                     break
                    
#                 y_start, y_end = gh * cell_h, (gh + 1) * cell_h
#                 x_start, x_end = gw * cell_w, (gw + 1) * cell_w
                
#                 cell_mask = (
#                     (bg_coords[:, 0] >= y_start) & (bg_coords[:, 0] < y_end) &
#                     (bg_coords[:, 1] >= x_start) & (bg_coords[:, 1] < x_end)
#                 )
#                 cell_coords = bg_coords[cell_mask]
                
#                 if len(cell_coords) > 0:
#                     cell_sims = np.array([sim_np[y, x] for y, x in cell_coords])
#                     min_idx = np.argmin(cell_sims)
#                     neg_points.append(cell_coords[min_idx])
        
#         if len(neg_points) < num_points:
#             remaining = num_points - len(neg_points)
#             if len(bg_coords) > 0:
#                 indices = np.random.choice(len(bg_coords), 
#                                           min(remaining, len(bg_coords)), 
#                                           replace=False)
#                 for idx in indices:
#                     neg_points.append(bg_coords[idx])
        
#         return np.array(neg_points[:num_points])
    
#     def _sample_from_borders(self, H, W, num_points):
#         border = 3
#         neg_points = []
        
#         corners = [[border, border], [border, W-border-1], 
#                    [H-border-1, border], [H-border-1, W-border-1]]
        
#         for corner in corners[:min(num_points, 4)]:
#             neg_points.append(corner)
        
#         while len(neg_points) < num_points:
#             side = np.random.randint(4)
#             if side == 0:
#                 neg_points.append([border, np.random.randint(border, W-border)])
#             elif side == 1:
#                 neg_points.append([H-border-1, np.random.randint(border, W-border)])
#             elif side == 2:
#                 neg_points.append([np.random.randint(border, H-border), border])
#             else:
#                 neg_points.append([np.random.randint(border, H-border), W-border-1])
        
#         return np.array(neg_points)
    
#     def generate_rough_mask(self, query_img, support_images, support_masks,
#                            sam2_predictor, top_k=10, neg_k=5):
#         query_features = self.extract_dino_features(query_img)
#         C, H_feat, W_feat = query_features.shape
        
#         support_features_list = []
#         support_masks_list = []
#         for sup_img, sup_mask in zip(support_images, support_masks):
#             sup_features = self.extract_dino_features(sup_img)
#             downsampled_mask = self.downsample_mask(sup_mask, (H_feat, W_feat))
#             support_features_list.append(sup_features)
#             support_masks_list.append(downsampled_mask)
        
#         # 使用Prototype方法计算相似度
#         similarity_map = self.compute_similarity_map_prototype(
#             query_features, support_features_list, support_masks_list
#         )
        
#         point_coords_feat, point_labels = self.get_prompts_from_similarity(
#             similarity_map, top_k=top_k, neg_k=neg_k
#         )
        
#         H_img, W_img = query_img.size[::-1]
#         scale_y = H_img / H_feat
#         scale_x = W_img / W_feat
        
#         point_coords_img = point_coords_feat.copy().astype(np.float32)
#         point_coords_img[:, 0] *= scale_x
#         point_coords_img[:, 1] *= scale_y
#         point_coords_img = point_coords_img.astype(np.int32)
        
#         sam2_predictor.set_image(np.array(query_img))
        
#         try:
#             # ===== Multimask输出 =====
#             masks, scores, _ = sam2_predictor.predict(
#                 point_coords=point_coords_img,
#                 point_labels=point_labels,
#                 multimask_output=True
#             )
#             best_idx = np.argmax(scores)
#             rough_mask = masks[best_idx]
            
#         except Exception as e:
#             print(f"   ⚠️  SAM2预测失败: {e}")
#             rough_mask = None
        
#         return rough_mask, similarity_map.cpu().numpy(), point_coords_img


# # Memory模块完全不变
# class SAM2MemoryModule:
#     def __init__(self, sam2_model, device='cuda', sam2_model_type='large'):
#         self.model = sam2_model
#         self.device = device
#         self.sam2_model_type = sam2_model_type
#         self.prev_out = {}
#         self.support_set = False
#         self.image_size = getattr(sam2_model, 'image_size', 1024)
#         self.num_feature_levels = getattr(sam2_model, 'num_feature_levels', 3)
    
#     def clear_support(self):
#         self.prev_out = {}
#         self.support_set = False
    
#     def _prepare_image(self, img):
#         target_size = self.image_size
#         img_resized = img.resize((target_size, target_size), Image.BILINEAR)
#         img_np = np.array(img_resized).astype(np.float32) / 255.0
#         img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
#         img_tensor = img_tensor.unsqueeze(0).to(self.device)
#         return img_tensor
    
#     def _prepare_mask(self, mask, target_size=None):
#         if target_size is None:
#             target_size = self.image_size
#         mask_resized = cv2.resize(mask.astype(np.float32), (target_size, target_size),
#                                   interpolation=cv2.INTER_NEAREST)
#         mask_tensor = torch.from_numpy(mask_resized).float()
#         mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
#         return mask_tensor
    
#     def encode_support(self, support_img, support_mask):
#         img_tensor = self._prepare_image(support_img)
        
#         with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
#             backbone_out = self.model.forward_image(img_tensor)
#             feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
#             vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels:]
            
#             for i in range(len(feature_maps)):
#                 feature_maps[i] = feature_maps[i].clone()
#                 vision_pos_embeds[i] = vision_pos_embeds[i].clone()
            
#             pix_feat = feature_maps[-1]
#             mask_tensor = self._prepare_mask(support_mask)
#             high_res_masks = mask_tensor.float() * 20.0 - 10.0
            
#             maskmem_out = self.model.memory_encoder(
#                 pix_feat=pix_feat, masks=high_res_masks, skip_mask_sigmoid=True
#             )
            
#             self.prev_out.setdefault("maskmem_features", []).append(
#                 maskmem_out["vision_features"].clone()
#             )
#             if "maskmem_pos_enc" not in self.prev_out:
#                 self.prev_out["maskmem_pos_enc"] = [
#                     m.clone() for m in maskmem_out["vision_pos_enc"]
#                 ]
        
#         self.support_set = True
    
#     def set_support(self, support_images, support_masks, target_size=None):
#         self.clear_support()
#         for img, mask in zip(support_images, support_masks):
#             self.encode_support(img, mask)
#         return self.support_set
    
#     def predict_query(self, query_img, rough_mask=None):
#         if not self.support_set:
#             raise ValueError("请先调用set_support()")
        
#         original_size = query_img.size[::-1]
#         img_tensor = self._prepare_image(query_img)
#         B = img_tensor.shape[0]
        
#         with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
#             backbone_out = self.model.forward_image(img_tensor)
#             feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
#             vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels:]
            
#             for i in range(len(feature_maps)):
#                 feature_maps[i] = feature_maps[i].clone()
#                 vision_pos_embeds[i] = vision_pos_embeds[i].clone()
            
#             high_res_features = feature_maps[:-1]
#             pix_feat = feature_maps[-1]
            
#             to_cat_memory = [m.flatten(2).permute(2, 0, 1) for m in self.prev_out["maskmem_features"]]
#             to_cat_memory_pos_embed = len(self.prev_out["maskmem_features"]) * [
#                 self.prev_out["maskmem_pos_enc"][0].flatten(2).permute(2, 0, 1)
#             ]
            
#             memory = torch.cat(to_cat_memory, dim=0)
#             memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)
            
#             pix_feat_with_mem = self.model.memory_attention(
#                 curr=pix_feat.flatten(2).permute(2, 0, 1),
#                 curr_pos=vision_pos_embeds[-1].flatten(2).permute(2, 0, 1),
#                 memory=memory, memory_pos=memory_pos_embed, num_obj_ptr_tokens=0
#             )
#             pix_feat_with_mem = pix_feat_with_mem.clone().permute(1, 2, 0).view(*pix_feat.shape)
            
#             sam_point_coords = torch.zeros(B, 1, 2, device=self.device)
#             sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=self.device)
            
#             sam_mask_prompt = None
#             if rough_mask is not None:
#                 mask_tensor = self._prepare_mask(rough_mask, target_size=self.image_size)
#                 high_res_mask_logits = mask_tensor.float() * 20.0 - 10.0
#                 sam_mask_prompt = F.interpolate(high_res_mask_logits, size=(256, 256),
#                                                 mode='bilinear', align_corners=False)
            
#             sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
#                 points=(sam_point_coords, sam_point_labels), boxes=None, masks=sam_mask_prompt
#             )
#             sparse_embeddings = sparse_embeddings.clone()
#             dense_embeddings = dense_embeddings.clone()
#             image_pe = self.model.sam_prompt_encoder.get_dense_pe().clone()
            
#             low_res_masks, ious, _, _ = self.model.sam_mask_decoder(
#                 image_embeddings=pix_feat_with_mem, image_pe=image_pe,
#                 sparse_prompt_embeddings=sparse_embeddings,
#                 dense_prompt_embeddings=dense_embeddings,
#                 multimask_output=False, repeat_image=False, high_res_features=high_res_features
#             )
            
#             low_res_masks = low_res_masks.float()
#             high_res_masks = F.interpolate(low_res_masks, size=(self.image_size, self.image_size),
#                                            mode='bilinear', align_corners=False)
#             final_masks = F.interpolate(high_res_masks, size=original_size,
#                                         mode='bilinear', align_corners=False)
            
#             pred_mask = (final_masks[0, 0] > 0).cpu().numpy().astype(np.uint8)
#             score = float(ious[0, 0].cpu())
        
#         return pred_mask, score


# class SPSAMModel:
#     def __init__(self, sam2_model, sam2_predictor, dino_model, dino_transform,
#                  device='cuda', sam2_model_type='large'):
#         self.sam2_model = sam2_model
#         self.sam2_predictor = sam2_predictor
#         self.device = device
        
#         self.cmrs = CMRSModule(dino_model, dino_transform, device)
#         self.memory_module = SAM2MemoryModule(sam2_model, device, sam2_model_type)
        
#         self._support_images = []
#         self._support_masks = []
#         self._support_set = False
        
#         print("✅ SPSAMModel (Multimask + Prototype)")
    
#     def set_support(self, support_images, support_masks):
#         self._support_images = support_images
#         self._support_masks = support_masks
#         success = self.memory_module.set_support(support_images, support_masks)
#         self._support_set = success
#         return success
    
#     def clear_support(self):
#         self._support_images = []
#         self._support_masks = []
#         self._support_set = False
#         self.memory_module.clear_support()
    
#     def predict(self, query_img, support_images, support_masks,
#                use_cmrs=True, use_memory_refinement=False):
#         if use_memory_refinement:
#             self.set_support(support_images, support_masks)
#         return self.predict_query(query_img, use_cmrs, use_memory_refinement,
#                                   support_images, support_masks)
    
#     def predict_query(self, query_img, use_cmrs=True, use_memory_refinement=False,
#                      support_images=None, support_masks=None):
#         sup_imgs = support_images if support_images else self._support_images
#         sup_masks = support_masks if support_masks else self._support_masks
        
#         if len(sup_imgs) == 0:
#             raise ValueError("No support samples")
        
#         results = {'final_mask': None, 'rough_mask': None, 'similarity_map': None, 'score': 0.0}
        
#         rough_mask = None
#         if use_cmrs:
#             rough_mask, similarity_map, _ = self.cmrs.generate_rough_mask(
#                 query_img, sup_imgs, sup_masks, self.sam2_predictor
#             )
#             results['rough_mask'] = rough_mask
#             results['similarity_map'] = similarity_map
        
#         if use_memory_refinement:
#             if not self._support_set:
#                 self.set_support(sup_imgs, sup_masks)
#             pred_mask, score = self.memory_module.predict_query(
#                 query_img, rough_mask=rough_mask if use_cmrs else None
#             )
#             results['final_mask'] = pred_mask
#             results['score'] = score
#         else:
#             if rough_mask is not None:
#                 results['final_mask'] = rough_mask.astype(np.uint8)
#                 results['score'] = 1.0
#             else:
#                 H, W = query_img.size[::-1]
#                 results['final_mask'] = np.zeros((H, W), dtype=np.uint8)
#                 results['score'] = 0.0
        
#         return results


# def visualize_sp_sam_results(query_img, gt_mask, pred_results,
#                             support_images=None, support_masks=None,
#                             save_path=None, title="SP-SAM Results"):
#     import matplotlib.pyplot as plt
    
#     n_cols = 4
#     n_rows = 1 + (1 if support_images else 0)
    
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
#     if n_rows == 1:
#         axes = [axes]
    
#     row = axes[0] if n_rows > 1 else axes
    
#     row[0].imshow(query_img)
#     row[0].set_title("Query")
#     row[0].axis('off')
    
#     row[1].imshow(gt_mask, cmap='gray')
#     row[1].set_title("GT")
#     row[1].axis('off')
    
#     rough_mask = pred_results.get('rough_mask')
#     if rough_mask is not None:
#         row[2].imshow(rough_mask, cmap='gray')
#         row[2].set_title("Rough")
#     row[2].axis('off')
    
#     final_mask = pred_results.get('final_mask')
#     if final_mask is not None:
#         row[3].imshow(final_mask, cmap='gray')
#         if gt_mask is not None:
#             intersection = (final_mask > 0) & (gt_mask > 0)
#             union = (final_mask > 0) | (gt_mask > 0)
#             iou = intersection.sum() / (union.sum() + 1e-8)
#             row[3].set_title(f"Final (IoU: {iou:.3f})")
#     row[3].axis('off')
    
#     plt.suptitle(title)
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         plt.close()
#     else:
#         plt.show()


# if __name__ == '__main__':
#     print("SP-SAM Multimask + Prototype 组合版本")



# """
# SP-SAM 消融实验版本
# ====================

# 实验目的：分离Memory Attention和Mask Prompt的作用

# 实验设计：
# - 实验1 (mask_only): 只用Mask Prompt，不用Memory Attention
# - 实验2 (memory_only_no_mask): 只用Memory Attention，不用Mask Prompt  
# - 实验3 (both): 两者都用 (当前默认)
# - 实验4 (neither): 两者都不用 (对照)

# 修改底部的 ABLATION_MODE 来选择实验
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import cv2
# from PIL import Image
# from typing import List, Dict, Tuple, Optional, Any


# class CMRSModule:
#     """CMRS模块 - 保持不变"""
    
#     def __init__(self, dino_model, dino_transform, device='cuda'):
#         self.dino_model = dino_model
#         self.dino_transform = dino_transform
#         self.device = device
        
#     def extract_dino_features(self, img_pil):
#         img_tensor = self.dino_transform(img_pil)[None].to(self.device)
#         with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
#             features = self.dino_model.get_intermediate_layers(img_tensor.to(torch.bfloat16))[0]
#             h = w = int(features.shape[1]**0.5)
#             feature_map = features.reshape(1, h, w, -1).permute(0, 3, 1, 2).float()
#         return feature_map.squeeze(0)
    
#     def downsample_mask(self, mask_np, target_size):
#         h, w = target_size
#         resized_mask = cv2.resize(mask_np.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
#         resized_mask = (resized_mask > 0.5).astype(np.float32)
#         return torch.from_numpy(resized_mask).to(self.device)
    
#     def compute_prototype(self, features, mask):
#         mask_sum = mask.sum() + 1e-8
#         prototype = (features * mask.unsqueeze(0)).sum(dim=(1, 2)) / mask_sum
#         return prototype
    
#     def compute_similarity_map_prototype(self, query_features, support_features_list, support_masks_list):
#         C, H, W = query_features.shape
#         prototypes = []
#         for sup_feat, sup_mask in zip(support_features_list, support_masks_list):
#             proto = self.compute_prototype(sup_feat, sup_mask)
#             prototypes.append(proto)
#         prototypes = torch.stack(prototypes, dim=0)
#         avg_prototype = prototypes.mean(dim=0)
#         avg_prototype = F.normalize(avg_prototype, p=2, dim=0)
#         query_flat = query_features.reshape(C, -1).T
#         query_flat = F.normalize(query_flat, p=2, dim=1)
#         similarity_scores = torch.mv(query_flat, avg_prototype)
#         return similarity_scores.reshape(H, W)
    
#     def get_prompts_from_similarity(self, similarity_map, top_k=10, neg_k=5):
#         sim_np = similarity_map.cpu().numpy()
#         H, W = sim_np.shape
#         mean_sim, std_sim, max_sim = sim_np.mean(), sim_np.std(), sim_np.max()
        
#         fg_threshold = max(mean_sim + 1.0 * std_sim, max_sim * 0.5)
#         fg_mask = sim_np > fg_threshold
#         fg_coords = np.argwhere(fg_mask)
        
#         if len(fg_coords) < top_k:
#             fg_threshold = max(mean_sim + 0.5 * std_sim, max_sim * 0.4)
#             fg_coords = np.argwhere(sim_np > fg_threshold)
#         if len(fg_coords) < top_k:
#             fg_coords = np.argwhere(sim_np > mean_sim + 0.3 * std_sim)
        
#         if len(fg_coords) == 0:
#             flat_indices = np.argsort(sim_np.flatten())[-top_k:]
#             pos_points = np.array([np.unravel_index(idx, sim_np.shape) for idx in flat_indices])
#         else:
#             pos_points = self._sample_points_weighted_spatial(fg_coords, sim_np, top_k)
        
#         bg_threshold = mean_sim - 0.5 * std_sim
#         bg_coords = np.argwhere(sim_np < bg_threshold)
        
#         if len(bg_coords) >= neg_k:
#             neg_points = self._sample_negative_points_diverse(bg_coords, sim_np, neg_k, (H, W))
#         else:
#             neg_points = self._sample_from_borders(H, W, neg_k)
        
#         all_points = np.vstack([pos_points, neg_points])
#         all_labels = np.concatenate([np.ones(len(pos_points), dtype=np.int32),
#                                      np.zeros(len(neg_points), dtype=np.int32)])
#         return all_points[:, ::-1], all_labels
    
#     def _sample_points_weighted_spatial(self, coords, sim_np, num_points):
#         sim_values = np.array([sim_np[y, x] for y, x in coords])
#         selected, selected_indices = [], []
#         first_idx = np.argmax(sim_values)
#         selected.append(coords[first_idx])
#         selected_indices.append(first_idx)
        
#         for _ in range(num_points - 1):
#             if len(selected_indices) >= len(coords):
#                 break
#             best_score, best_idx = -np.inf, -1
#             for i, (coord, sim_val) in enumerate(zip(coords, sim_values)):
#                 if i in selected_indices:
#                     continue
#                 min_dist = min(np.sqrt((coord[0]-s[0])**2 + (coord[1]-s[1])**2) for s in selected)
#                 score = sim_val * 0.7 + (min_dist / max(sim_np.shape)) * 0.3
#                 if score > best_score:
#                     best_score, best_idx = score, i
#             if best_idx >= 0:
#                 selected.append(coords[best_idx])
#                 selected_indices.append(best_idx)
#         return np.array(selected)
    
#     def _sample_negative_points_diverse(self, bg_coords, sim_np, num_points, shape):
#         H, W = shape
#         grid_h, grid_w = 3, 3
#         cell_h, cell_w = H // grid_h, W // grid_w
#         neg_points = []
#         for gh in range(grid_h):
#             for gw in range(grid_w):
#                 if len(neg_points) >= num_points:
#                     break
#                 y_start, y_end = gh * cell_h, (gh + 1) * cell_h
#                 x_start, x_end = gw * cell_w, (gw + 1) * cell_w
#                 cell_mask = ((bg_coords[:, 0] >= y_start) & (bg_coords[:, 0] < y_end) &
#                             (bg_coords[:, 1] >= x_start) & (bg_coords[:, 1] < x_end))
#                 cell_coords = bg_coords[cell_mask]
#                 if len(cell_coords) > 0:
#                     cell_sims = np.array([sim_np[y, x] for y, x in cell_coords])
#                     neg_points.append(cell_coords[np.argmin(cell_sims)])
#         if len(neg_points) < num_points and len(bg_coords) > 0:
#             indices = np.random.choice(len(bg_coords), min(num_points - len(neg_points), len(bg_coords)), replace=False)
#             for idx in indices:
#                 neg_points.append(bg_coords[idx])
#         return np.array(neg_points[:num_points])
    
#     def _sample_from_borders(self, H, W, num_points):
#         border = 3
#         corners = [[border, border], [border, W-border-1], [H-border-1, border], [H-border-1, W-border-1]]
#         neg_points = corners[:min(num_points, 4)]
#         while len(neg_points) < num_points:
#             side = np.random.randint(4)
#             if side == 0:
#                 neg_points.append([border, np.random.randint(border, W-border)])
#             elif side == 1:
#                 neg_points.append([H-border-1, np.random.randint(border, W-border)])
#             elif side == 2:
#                 neg_points.append([np.random.randint(border, H-border), border])
#             else:
#                 neg_points.append([np.random.randint(border, H-border), W-border-1])
#         return np.array(neg_points)
    
#     def generate_rough_mask(self, query_img, support_images, support_masks, sam2_predictor, top_k=10, neg_k=5):
#         query_features = self.extract_dino_features(query_img)
#         C, H_feat, W_feat = query_features.shape
        
#         support_features_list, support_masks_list = [], []
#         for sup_img, sup_mask in zip(support_images, support_masks):
#             sup_features = self.extract_dino_features(sup_img)
#             downsampled_mask = self.downsample_mask(sup_mask, (H_feat, W_feat))
#             support_features_list.append(sup_features)
#             support_masks_list.append(downsampled_mask)
        
#         similarity_map = self.compute_similarity_map_prototype(query_features, support_features_list, support_masks_list)
#         point_coords_feat, point_labels = self.get_prompts_from_similarity(similarity_map, top_k, neg_k)
        
#         H_img, W_img = query_img.size[::-1]
#         scale_y, scale_x = H_img / H_feat, W_img / W_feat
        
#         point_coords_img = point_coords_feat.copy().astype(np.float32)
#         point_coords_img[:, 0] *= scale_x
#         point_coords_img[:, 1] *= scale_y
#         point_coords_img = point_coords_img.astype(np.int32)
        
#         sam2_predictor.set_image(np.array(query_img))
        
#         try:
#             masks, scores, _ = sam2_predictor.predict(
#                 point_coords=point_coords_img, point_labels=point_labels, multimask_output=True
#             )
#             rough_mask = masks[np.argmax(scores)]
#         except Exception as e:
#             print(f"   ⚠️  SAM2预测失败: {e}")
#             rough_mask = None
        
#         return rough_mask, similarity_map.cpu().numpy(), point_coords_img


# # ============================================================
# # 🔬 消融实验版Memory模块
# # ============================================================

# # 选择实验模式
# ABLATION_MODE = 'memory_only_no_mask'  # ⬅️ 修改这里！

# # 可选值:
# # 'mask_only'          - 只用Mask Prompt，禁用Memory Attention
# # 'memory_only_no_mask' - 只用Memory Attention，不用Mask Prompt
# # 'both'               - 两者都用 (原始设计)
# # 'neither'            - 两者都不用 (纯SAM2 decoder)


# class SAM2MemoryModule:
#     """消融实验版Memory模块"""
    
#     def __init__(self, sam2_model, device='cuda', sam2_model_type='large'):
#         self.model = sam2_model
#         self.device = device
#         self.sam2_model_type = sam2_model_type
#         self.prev_out = {}
#         self.support_set = False
#         self.image_size = getattr(sam2_model, 'image_size', 1024)
#         self.num_feature_levels = getattr(sam2_model, 'num_feature_levels', 3)
        
#         print(f"📌 消融实验模式: {ABLATION_MODE}")
#         if ABLATION_MODE == 'mask_only':
#             print("   → 只用Mask Prompt，禁用Memory Attention")
#         elif ABLATION_MODE == 'memory_only_no_mask':
#             print("   → 只用Memory Attention，不用Mask Prompt")
#         elif ABLATION_MODE == 'both':
#             print("   → 两者都用 (原始设计)")
#         elif ABLATION_MODE == 'neither':
#             print("   → 两者都不用 (纯SAM2 decoder)")
    
#     def clear_support(self):
#         self.prev_out = {}
#         self.support_set = False
    
#     def _prepare_image(self, img):
#         target_size = self.image_size
#         img_resized = img.resize((target_size, target_size), Image.BILINEAR)
#         img_np = np.array(img_resized).astype(np.float32) / 255.0
#         img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
#         return img_tensor.unsqueeze(0).to(self.device)
    
#     def _prepare_mask(self, mask, target_size=None):
#         if target_size is None:
#             target_size = self.image_size
#         mask_resized = cv2.resize(mask.astype(np.float32), (target_size, target_size), interpolation=cv2.INTER_NEAREST)
#         mask_tensor = torch.from_numpy(mask_resized).float()
#         return mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
    
#     def encode_support(self, support_img, support_mask):
#         img_tensor = self._prepare_image(support_img)
        
#         with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
#             backbone_out = self.model.forward_image(img_tensor)
#             feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
            
#             for i in range(len(feature_maps)):
#                 feature_maps[i] = feature_maps[i].clone()
            
#             pix_feat = feature_maps[-1]
#             mask_tensor = self._prepare_mask(support_mask)
#             high_res_masks = mask_tensor.float() * 20.0 - 10.0
            
#             maskmem_out = self.model.memory_encoder(pix_feat=pix_feat, masks=high_res_masks, skip_mask_sigmoid=True)
            
#             self.prev_out.setdefault("maskmem_features", []).append(maskmem_out["vision_features"].clone())
#             if "maskmem_pos_enc" not in self.prev_out:
#                 self.prev_out["maskmem_pos_enc"] = [m.clone() for m in maskmem_out["vision_pos_enc"]]
        
#         self.support_set = True
    
#     def set_support(self, support_images, support_masks, target_size=None):
#         self.clear_support()
#         for img, mask in zip(support_images, support_masks):
#             self.encode_support(img, mask)
#         return self.support_set
    
#     def predict_query(self, query_img, rough_mask=None):
#         if not self.support_set:
#             raise ValueError("请先调用set_support()")
        
#         original_size = query_img.size[::-1]
#         img_tensor = self._prepare_image(query_img)
#         B = img_tensor.shape[0]
        
#         with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
#             backbone_out = self.model.forward_image(img_tensor)
#             feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
#             vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels:]
            
#             for i in range(len(feature_maps)):
#                 feature_maps[i] = feature_maps[i].clone()
#                 vision_pos_embeds[i] = vision_pos_embeds[i].clone()
            
#             high_res_features = feature_maps[:-1]
#             pix_feat = feature_maps[-1]
            
#             # ============================================================
#             # 🔬 消融点1: Memory Attention
#             # ============================================================
#             if ABLATION_MODE in ['both', 'memory_only_no_mask']:
#                 # 使用Memory Attention
#                 to_cat_memory = [m.flatten(2).permute(2, 0, 1) for m in self.prev_out["maskmem_features"]]
#                 to_cat_memory_pos_embed = len(self.prev_out["maskmem_features"]) * [
#                     self.prev_out["maskmem_pos_enc"][0].flatten(2).permute(2, 0, 1)
#                 ]
                
#                 memory = torch.cat(to_cat_memory, dim=0)
#                 memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)
                
#                 pix_feat_with_mem = self.model.memory_attention(
#                     curr=pix_feat.flatten(2).permute(2, 0, 1),
#                     curr_pos=vision_pos_embeds[-1].flatten(2).permute(2, 0, 1),
#                     memory=memory, memory_pos=memory_pos_embed, num_obj_ptr_tokens=0
#                 )
#                 pix_feat_with_mem = pix_feat_with_mem.clone().permute(1, 2, 0).view(*pix_feat.shape)
#             else:
#                 # 禁用Memory Attention，直接使用原始特征
#                 pix_feat_with_mem = pix_feat
            
#             # 空的点提示
#             sam_point_coords = torch.zeros(B, 1, 2, device=self.device)
#             sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=self.device)
            
#             # ============================================================
#             # 🔬 消融点2: Mask Prompt
#             # ============================================================
#             sam_mask_prompt = None
#             if ABLATION_MODE in ['both', 'mask_only']:
#                 # 使用Mask Prompt
#                 if rough_mask is not None:
#                     mask_tensor = self._prepare_mask(rough_mask, target_size=self.image_size)
#                     high_res_mask_logits = mask_tensor.float() * 20.0 - 10.0
#                     sam_mask_prompt = F.interpolate(high_res_mask_logits, size=(256, 256),
#                                                     mode='bilinear', align_corners=False)
#             # else: sam_mask_prompt = None (不使用mask prompt)
            
#             sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
#                 points=(sam_point_coords, sam_point_labels), boxes=None, masks=sam_mask_prompt
#             )
#             sparse_embeddings = sparse_embeddings.clone()
#             dense_embeddings = dense_embeddings.clone()
#             image_pe = self.model.sam_prompt_encoder.get_dense_pe().clone()
            
#             low_res_masks, ious, _, _ = self.model.sam_mask_decoder(
#                 image_embeddings=pix_feat_with_mem, image_pe=image_pe,
#                 sparse_prompt_embeddings=sparse_embeddings,
#                 dense_prompt_embeddings=dense_embeddings,
#                 multimask_output=False, repeat_image=False, high_res_features=high_res_features
#             )
            
#             low_res_masks = low_res_masks.float()
#             high_res_masks = F.interpolate(low_res_masks, size=(self.image_size, self.image_size),
#                                            mode='bilinear', align_corners=False)
#             final_masks = F.interpolate(high_res_masks, size=original_size,
#                                         mode='bilinear', align_corners=False)
            
#             pred_mask = (final_masks[0, 0] > 0).cpu().numpy().astype(np.uint8)
#             score = float(ious[0, 0].cpu())
        
#         return pred_mask, score


# class SPSAMModel:
#     def __init__(self, sam2_model, sam2_predictor, dino_model, dino_transform,
#                  device='cuda', sam2_model_type='large'):
#         self.sam2_model = sam2_model
#         self.sam2_predictor = sam2_predictor
#         self.device = device
        
#         self.cmrs = CMRSModule(dino_model, dino_transform, device)
#         self.memory_module = SAM2MemoryModule(sam2_model, device, sam2_model_type)
        
#         self._support_images = []
#         self._support_masks = []
#         self._support_set = False
        
#         print(f"✅ SPSAMModel (消融实验版)")
    
#     def set_support(self, support_images, support_masks):
#         self._support_images = support_images
#         self._support_masks = support_masks
#         success = self.memory_module.set_support(support_images, support_masks)
#         self._support_set = success
#         return success
    
#     def clear_support(self):
#         self._support_images = []
#         self._support_masks = []
#         self._support_set = False
#         self.memory_module.clear_support()
    
#     def predict(self, query_img, support_images, support_masks, use_cmrs=True, use_memory_refinement=False):
#         if use_memory_refinement:
#             self.set_support(support_images, support_masks)
#         return self.predict_query(query_img, use_cmrs, use_memory_refinement, support_images, support_masks)
    
#     def predict_query(self, query_img, use_cmrs=True, use_memory_refinement=False,
#                      support_images=None, support_masks=None):
#         sup_imgs = support_images if support_images else self._support_images
#         sup_masks = support_masks if support_masks else self._support_masks
        
#         if len(sup_imgs) == 0:
#             raise ValueError("No support samples")
        
#         results = {'final_mask': None, 'rough_mask': None, 'similarity_map': None, 'score': 0.0}
        
#         rough_mask = None
#         if use_cmrs:
#             rough_mask, similarity_map, _ = self.cmrs.generate_rough_mask(
#                 query_img, sup_imgs, sup_masks, self.sam2_predictor
#             )
#             results['rough_mask'] = rough_mask
#             results['similarity_map'] = similarity_map
        
#         if use_memory_refinement:
#             if not self._support_set:
#                 self.set_support(sup_imgs, sup_masks)
#             pred_mask, score = self.memory_module.predict_query(
#                 query_img, rough_mask=rough_mask if use_cmrs else None
#             )
#             results['final_mask'] = pred_mask
#             results['score'] = score
#         else:
#             if rough_mask is not None:
#                 results['final_mask'] = rough_mask.astype(np.uint8)
#                 results['score'] = 1.0
#             else:
#                 H, W = query_img.size[::-1]
#                 results['final_mask'] = np.zeros((H, W), dtype=np.uint8)
#                 results['score'] = 0.0
        
#         return results


# def visualize_sp_sam_results(query_img, gt_mask, pred_results,
#                             support_images=None, support_masks=None,
#                             save_path=None, title="SP-SAM Results"):
#     import matplotlib.pyplot as plt
#     n_cols = 4
#     fig, axes = plt.subplots(1, n_cols, figsize=(16, 4))
    
#     axes[0].imshow(query_img)
#     axes[0].set_title("Query")
#     axes[0].axis('off')
    
#     axes[1].imshow(gt_mask, cmap='gray')
#     axes[1].set_title("GT")
#     axes[1].axis('off')
    
#     if pred_results.get('rough_mask') is not None:
#         axes[2].imshow(pred_results['rough_mask'], cmap='gray')
#         axes[2].set_title("Rough")
#     axes[2].axis('off')
    
#     if pred_results.get('final_mask') is not None:
#         axes[3].imshow(pred_results['final_mask'], cmap='gray')
#         if gt_mask is not None:
#             intersection = (pred_results['final_mask'] > 0) & (gt_mask > 0)
#             union = (pred_results['final_mask'] > 0) | (gt_mask > 0)
#             iou = intersection.sum() / (union.sum() + 1e-8)
#             axes[3].set_title(f"Final (IoU: {iou:.3f})")
#     axes[3].axis('off')
    
#     plt.suptitle(title)
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         plt.close()
#     else:
#         plt.show()


# if __name__ == '__main__':
#     print("=" * 60)
#     print("SP-SAM 消融实验")
#     print("=" * 60)
#     print(f"\n当前模式: {ABLATION_MODE}")
#     print("\n实验设计:")
#     print("  'mask_only'           → 只用Mask Prompt")
#     print("  'memory_only_no_mask' → 只用Memory Attention")
#     print("  'both'                → 两者都用 (原始)")
#     print("  'neither'             → 两者都不用")
#     print("\n修改 ABLATION_MODE 变量来切换实验")


# """
# SP-SAM 渐进式改进版本
# =====================

# 提供3个独立的小改动，逐个测试效果:

# 版本A: 只改 multimask=True (最保守)
# 版本B: 只传递点提示，保持mask强度不变
# 版本C: 只降低mask强度，不传点提示

# 使用方法: 修改底部的 USE_VERSION 变量选择版本
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import cv2
# from PIL import Image
# from typing import List, Dict, Tuple, Optional, Any
# from tqdm import tqdm


# # ============================================================
# # CMRS模块 (不变)
# # ============================================================
# class CMRSModule:
#     def __init__(self, dino_model, dino_transform, device='cuda'):
#         self.dino_model = dino_model
#         self.dino_transform = dino_transform
#         self.device = device
        
#     def extract_dino_features(self, img_pil: Image.Image) -> torch.Tensor:
#         img_tensor = self.dino_transform(img_pil)[None].to(self.device)
        
#         with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
#             features = self.dino_model.get_intermediate_layers(
#                 img_tensor.to(torch.bfloat16)
#             )[0]
            
#             h = w = int(features.shape[1]**0.5)
#             feature_map = features.reshape(1, h, w, -1).permute(0, 3, 1, 2)
#             feature_map = feature_map.float()
            
#         return feature_map.squeeze(0)
    
#     def downsample_mask(self, mask_np: np.ndarray, target_size: Tuple[int, int]) -> torch.Tensor:
#         h, w = target_size
#         resized_mask = cv2.resize(mask_np.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
#         resized_mask = (resized_mask > 0.5).astype(np.float32)
#         return torch.from_numpy(resized_mask).to(self.device)
    
#     def compute_prototype(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#         mask_sum = mask.sum() + 1e-8
#         prototype = (features * mask.unsqueeze(0)).sum(dim=(1, 2)) / mask_sum
#         return prototype
    
#     def compute_similarity_map_prototype(self, query_features, support_features_list, support_masks_list):
#         C, H, W = query_features.shape
        
#         prototypes = []
#         for sup_feat, sup_mask in zip(support_features_list, support_masks_list):
#             proto = self.compute_prototype(sup_feat, sup_mask)
#             prototypes.append(proto)
        
#         prototypes = torch.stack(prototypes, dim=0)
#         avg_prototype = prototypes.mean(dim=0)
#         avg_prototype = F.normalize(avg_prototype, p=2, dim=0)
        
#         query_flat = query_features.reshape(C, -1).T
#         query_flat = F.normalize(query_flat, p=2, dim=1)
        
#         similarity_scores = torch.mv(query_flat, avg_prototype)
#         similarity_map = similarity_scores.reshape(H, W)
#         return similarity_map
    
#     def get_prompts_from_similarity(self, similarity_map, top_k=10, neg_k=5):
#         sim_np = similarity_map.cpu().numpy()
#         H, W = sim_np.shape
        
#         mean_sim = sim_np.mean()
#         std_sim = sim_np.std()
#         max_sim = sim_np.max()
        
#         fg_threshold = max(mean_sim + 1.0 * std_sim, max_sim * 0.5)
#         fg_mask = sim_np > fg_threshold
#         fg_coords = np.argwhere(fg_mask)
        
#         if len(fg_coords) < top_k:
#             fg_threshold = max(mean_sim + 0.5 * std_sim, max_sim * 0.4)
#             fg_mask = sim_np > fg_threshold
#             fg_coords = np.argwhere(fg_mask)
        
#         if len(fg_coords) < top_k:
#             fg_threshold = mean_sim + 0.3 * std_sim
#             fg_mask = sim_np > fg_threshold
#             fg_coords = np.argwhere(fg_mask)
        
#         if len(fg_coords) == 0:
#             flat_indices = np.argsort(sim_np.flatten())[-top_k:]
#             pos_points = np.array([np.unravel_index(idx, sim_np.shape) for idx in flat_indices])
#         else:
#             pos_points = self._sample_points_weighted_spatial(fg_coords, sim_np, top_k)
        
#         bg_threshold = mean_sim - 0.5 * std_sim
#         bg_mask = sim_np < bg_threshold
#         bg_coords = np.argwhere(bg_mask)
        
#         if len(bg_coords) >= neg_k:
#             neg_points = self._sample_negative_points_diverse(bg_coords, sim_np, neg_k, (H, W))
#         else:
#             neg_points = self._sample_from_borders(H, W, neg_k)
        
#         all_points = np.vstack([pos_points, neg_points])
#         pos_labels = np.ones(len(pos_points), dtype=np.int32)
#         neg_labels = np.zeros(len(neg_points), dtype=np.int32)
#         all_labels = np.concatenate([pos_labels, neg_labels])
        
#         point_coords = all_points[:, ::-1]
#         return point_coords, all_labels
    
#     def _sample_points_weighted_spatial(self, coords, sim_np, num_points):
#         sim_values = np.array([sim_np[y, x] for y, x in coords])
#         selected = []
#         selected_indices = []
        
#         first_idx = np.argmax(sim_values)
#         selected.append(coords[first_idx])
#         selected_indices.append(first_idx)
        
#         for _ in range(num_points - 1):
#             if len(selected_indices) >= len(coords):
#                 break
#             best_score = -np.inf
#             best_idx = -1
            
#             for i, (coord, sim_val) in enumerate(zip(coords, sim_values)):
#                 if i in selected_indices:
#                     continue
#                 min_dist = np.inf
#                 for sel_coord in selected:
#                     dist = np.sqrt((coord[0] - sel_coord[0])**2 + (coord[1] - sel_coord[1])**2)
#                     min_dist = min(min_dist, dist)
#                 score = sim_val * 0.7 + (min_dist / max(sim_np.shape)) * 0.3
#                 if score > best_score:
#                     best_score = score
#                     best_idx = i
            
#             if best_idx >= 0:
#                 selected.append(coords[best_idx])
#                 selected_indices.append(best_idx)
        
#         return np.array(selected)
    
#     def _sample_negative_points_diverse(self, bg_coords, sim_np, num_points, shape):
#         H, W = shape
#         grid_h, grid_w = 3, 3
#         cell_h, cell_w = H // grid_h, W // grid_w
#         neg_points = []
        
#         for gh in range(grid_h):
#             for gw in range(grid_w):
#                 if len(neg_points) >= num_points:
#                     break
#                 y_start, y_end = gh * cell_h, (gh + 1) * cell_h
#                 x_start, x_end = gw * cell_w, (gw + 1) * cell_w
#                 cell_mask = ((bg_coords[:, 0] >= y_start) & (bg_coords[:, 0] < y_end) &
#                             (bg_coords[:, 1] >= x_start) & (bg_coords[:, 1] < x_end))
#                 cell_coords = bg_coords[cell_mask]
#                 if len(cell_coords) > 0:
#                     cell_sims = np.array([sim_np[y, x] for y, x in cell_coords])
#                     min_idx = np.argmin(cell_sims)
#                     neg_points.append(cell_coords[min_idx])
        
#         if len(neg_points) < num_points:
#             remaining = num_points - len(neg_points)
#             if len(bg_coords) > 0:
#                 indices = np.random.choice(len(bg_coords), min(remaining, len(bg_coords)), replace=False)
#                 for idx in indices:
#                     neg_points.append(bg_coords[idx])
        
#         return np.array(neg_points[:num_points])
    
#     def _sample_from_borders(self, H, W, num_points):
#         border = 3
#         neg_points = []
#         corners = [[border, border], [border, W-border-1], [H-border-1, border], [H-border-1, W-border-1]]
#         for corner in corners[:min(num_points, 4)]:
#             neg_points.append(corner)
#         while len(neg_points) < num_points:
#             side = np.random.randint(4)
#             if side == 0:
#                 neg_points.append([border, np.random.randint(border, W-border)])
#             elif side == 1:
#                 neg_points.append([H-border-1, np.random.randint(border, W-border)])
#             elif side == 2:
#                 neg_points.append([np.random.randint(border, H-border), border])
#             else:
#                 neg_points.append([np.random.randint(border, H-border), W-border-1])
#         return np.array(neg_points)
    
#     def generate_rough_mask(self, query_img, support_images, support_masks, sam2_predictor, top_k=10, neg_k=5):
#         query_features = self.extract_dino_features(query_img)
#         C, H_feat, W_feat = query_features.shape
        
#         support_features_list = []
#         support_masks_list = []
#         for sup_img, sup_mask in zip(support_images, support_masks):
#             sup_features = self.extract_dino_features(sup_img)
#             downsampled_mask = self.downsample_mask(sup_mask, (H_feat, W_feat))
#             support_features_list.append(sup_features)
#             support_masks_list.append(downsampled_mask)
        
#         similarity_map = self.compute_similarity_map_prototype(query_features, support_features_list, support_masks_list)
#         point_coords_feat, point_labels = self.get_prompts_from_similarity(similarity_map, top_k=top_k, neg_k=neg_k)
        
#         H_img, W_img = query_img.size[::-1]
#         scale_y = H_img / H_feat
#         scale_x = W_img / W_feat
        
#         point_coords_img = point_coords_feat.copy().astype(np.float32)
#         point_coords_img[:, 0] *= scale_x
#         point_coords_img[:, 1] *= scale_y
#         point_coords_img = point_coords_img.astype(np.int32)
        
#         sam2_predictor.set_image(np.array(query_img))
        
#         try:
#             masks, scores, _ = sam2_predictor.predict(
#                 point_coords=point_coords_img,
#                 point_labels=point_labels,
#                 multimask_output=True
#             )
#             best_idx = np.argmax(scores)
#             rough_mask = masks[best_idx]
#         except Exception as e:
#             print(f"   ⚠️  SAM2预测失败: {e}")
#             rough_mask = None
        
#         return rough_mask, similarity_map.cpu().numpy(), point_coords_img, point_labels


# # ============================================================
# # 版本A: 只改 multimask=True (最保守)
# # ============================================================
# class SAM2MemoryModule_VersionA:
#     """只改multimask=True，其他完全不变"""
    
#     def __init__(self, sam2_model, device='cuda', sam2_model_type='large'):
#         self.model = sam2_model
#         self.device = device
#         self.sam2_model_type = sam2_model_type
#         self.prev_out = {}
#         self.support_set = False
#         self.image_size = getattr(sam2_model, 'image_size', 1024)
#         self.num_feature_levels = getattr(sam2_model, 'num_feature_levels', 3)
    
#     def clear_support(self):
#         self.prev_out = {}
#         self.support_set = False
    
#     def _prepare_image(self, img):
#         target_size = self.image_size
#         img_resized = img.resize((target_size, target_size), Image.BILINEAR)
#         img_np = np.array(img_resized).astype(np.float32) / 255.0
#         img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
#         img_tensor = img_tensor.unsqueeze(0).to(self.device)
#         return img_tensor
    
#     def _prepare_mask(self, mask, target_size=None):
#         if target_size is None:
#             target_size = self.image_size
#         mask_resized = cv2.resize(mask.astype(np.float32), (target_size, target_size), interpolation=cv2.INTER_NEAREST)
#         mask_tensor = torch.from_numpy(mask_resized).float()
#         mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
#         return mask_tensor
    
#     def encode_support(self, support_img, support_mask):
#         img_tensor = self._prepare_image(support_img)
        
#         with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
#             backbone_out = self.model.forward_image(img_tensor)
#             feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
#             for i in range(len(feature_maps)):
#                 feature_maps[i] = feature_maps[i].clone()
            
#             pix_feat = feature_maps[-1]
#             mask_tensor = self._prepare_mask(support_mask)
#             high_res_masks = mask_tensor.float() * 20.0 - 10.0
            
#             maskmem_out = self.model.memory_encoder(pix_feat=pix_feat, masks=high_res_masks, skip_mask_sigmoid=True)
            
#             self.prev_out.setdefault("maskmem_features", []).append(maskmem_out["vision_features"].clone())
#             if "maskmem_pos_enc" not in self.prev_out:
#                 self.prev_out["maskmem_pos_enc"] = [m.clone() for m in maskmem_out["vision_pos_enc"]]
        
#         self.support_set = True
    
#     def set_support(self, support_images, support_masks, target_size=None):
#         self.clear_support()
#         for img, mask in zip(support_images, support_masks):
#             self.encode_support(img, mask)
#         return self.support_set
    
#     def predict_query(self, query_img, rough_mask=None, point_coords=None, point_labels=None):
#         if not self.support_set:
#             raise ValueError("请先调用set_support()")
        
#         original_size = query_img.size[::-1]
#         img_tensor = self._prepare_image(query_img)
#         B = img_tensor.shape[0]
        
#         with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
#             backbone_out = self.model.forward_image(img_tensor)
#             feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
#             vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels:]
            
#             for i in range(len(feature_maps)):
#                 feature_maps[i] = feature_maps[i].clone()
#                 vision_pos_embeds[i] = vision_pos_embeds[i].clone()
            
#             high_res_features = feature_maps[:-1]
#             pix_feat = feature_maps[-1]
            
#             to_cat_memory = [m.flatten(2).permute(2, 0, 1) for m in self.prev_out["maskmem_features"]]
#             to_cat_memory_pos_embed = len(self.prev_out["maskmem_features"]) * [
#                 self.prev_out["maskmem_pos_enc"][0].flatten(2).permute(2, 0, 1)
#             ]
            
#             memory = torch.cat(to_cat_memory, dim=0)
#             memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)
            
#             pix_feat_with_mem = self.model.memory_attention(
#                 curr=pix_feat.flatten(2).permute(2, 0, 1),
#                 curr_pos=vision_pos_embeds[-1].flatten(2).permute(2, 0, 1),
#                 memory=memory, memory_pos=memory_pos_embed, num_obj_ptr_tokens=0
#             )
#             pix_feat_with_mem = pix_feat_with_mem.clone().permute(1, 2, 0).view(*pix_feat.shape)
            
#             # 保持原样：空点
#             sam_point_coords = torch.zeros(B, 1, 2, device=self.device)
#             sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=self.device)
            
#             # 保持原样：mask prompt强度
#             sam_mask_prompt = None
#             if rough_mask is not None:
#                 mask_tensor = self._prepare_mask(rough_mask, target_size=self.image_size)
#                 high_res_mask_logits = mask_tensor.float() * 20.0 - 10.0  # 原始强度
#                 sam_mask_prompt = F.interpolate(high_res_mask_logits, size=(256, 256), mode='bilinear', align_corners=False)
            
#             sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
#                 points=(sam_point_coords, sam_point_labels), boxes=None, masks=sam_mask_prompt
#             )
#             sparse_embeddings = sparse_embeddings.clone()
#             dense_embeddings = dense_embeddings.clone()
#             image_pe = self.model.sam_prompt_encoder.get_dense_pe().clone()
            
#             # ✅ 唯一改动: multimask_output=True
#             low_res_masks, ious, _, _ = self.model.sam_mask_decoder(
#                 image_embeddings=pix_feat_with_mem, image_pe=image_pe,
#                 sparse_prompt_embeddings=sparse_embeddings,
#                 dense_prompt_embeddings=dense_embeddings,
#                 multimask_output=True,  # ✅ 改动点
#                 repeat_image=False, high_res_features=high_res_features
#             )
            
#             # 选择最佳mask
#             best_idx = torch.argmax(ious[0])
#             best_mask = low_res_masks[0, best_idx:best_idx+1].unsqueeze(0)
#             best_iou = ious[0, best_idx]
            
#             best_mask = best_mask.float()
#             high_res_masks = F.interpolate(best_mask, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
#             final_masks = F.interpolate(high_res_masks, size=original_size, mode='bilinear', align_corners=False)
            
#             pred_mask = (final_masks[0, 0] > 0).cpu().numpy().astype(np.uint8)
#             score = float(best_iou.cpu())
        
#         return pred_mask, score


# # ============================================================
# # 版本B: 传递点提示，但保持mask强度不变
# # ============================================================
# class SAM2MemoryModule_VersionB:
#     """传递点提示，mask强度保持原样"""
    
#     def __init__(self, sam2_model, device='cuda', sam2_model_type='large'):
#         self.model = sam2_model
#         self.device = device
#         self.sam2_model_type = sam2_model_type
#         self.prev_out = {}
#         self.support_set = False
#         self.image_size = getattr(sam2_model, 'image_size', 1024)
#         self.num_feature_levels = getattr(sam2_model, 'num_feature_levels', 3)
    
#     def clear_support(self):
#         self.prev_out = {}
#         self.support_set = False
    
#     def _prepare_image(self, img):
#         target_size = self.image_size
#         img_resized = img.resize((target_size, target_size), Image.BILINEAR)
#         img_np = np.array(img_resized).astype(np.float32) / 255.0
#         img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
#         img_tensor = img_tensor.unsqueeze(0).to(self.device)
#         return img_tensor
    
#     def _prepare_mask(self, mask, target_size=None):
#         if target_size is None:
#             target_size = self.image_size
#         mask_resized = cv2.resize(mask.astype(np.float32), (target_size, target_size), interpolation=cv2.INTER_NEAREST)
#         mask_tensor = torch.from_numpy(mask_resized).float()
#         mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
#         return mask_tensor
    
#     def encode_support(self, support_img, support_mask):
#         img_tensor = self._prepare_image(support_img)
        
#         with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
#             backbone_out = self.model.forward_image(img_tensor)
#             feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
#             for i in range(len(feature_maps)):
#                 feature_maps[i] = feature_maps[i].clone()
            
#             pix_feat = feature_maps[-1]
#             mask_tensor = self._prepare_mask(support_mask)
#             high_res_masks = mask_tensor.float() * 20.0 - 10.0
            
#             maskmem_out = self.model.memory_encoder(pix_feat=pix_feat, masks=high_res_masks, skip_mask_sigmoid=True)
            
#             self.prev_out.setdefault("maskmem_features", []).append(maskmem_out["vision_features"].clone())
#             if "maskmem_pos_enc" not in self.prev_out:
#                 self.prev_out["maskmem_pos_enc"] = [m.clone() for m in maskmem_out["vision_pos_enc"]]
        
#         self.support_set = True
    
#     def set_support(self, support_images, support_masks, target_size=None):
#         self.clear_support()
#         for img, mask in zip(support_images, support_masks):
#             self.encode_support(img, mask)
#         return self.support_set
    
#     def predict_query(self, query_img, rough_mask=None, point_coords=None, point_labels=None):
#         if not self.support_set:
#             raise ValueError("请先调用set_support()")
        
#         original_size = query_img.size[::-1]
#         img_tensor = self._prepare_image(query_img)
#         B = img_tensor.shape[0]
        
#         with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
#             backbone_out = self.model.forward_image(img_tensor)
#             feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
#             vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels:]
            
#             for i in range(len(feature_maps)):
#                 feature_maps[i] = feature_maps[i].clone()
#                 vision_pos_embeds[i] = vision_pos_embeds[i].clone()
            
#             high_res_features = feature_maps[:-1]
#             pix_feat = feature_maps[-1]
            
#             to_cat_memory = [m.flatten(2).permute(2, 0, 1) for m in self.prev_out["maskmem_features"]]
#             to_cat_memory_pos_embed = len(self.prev_out["maskmem_features"]) * [
#                 self.prev_out["maskmem_pos_enc"][0].flatten(2).permute(2, 0, 1)
#             ]
            
#             memory = torch.cat(to_cat_memory, dim=0)
#             memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)
            
#             pix_feat_with_mem = self.model.memory_attention(
#                 curr=pix_feat.flatten(2).permute(2, 0, 1),
#                 curr_pos=vision_pos_embeds[-1].flatten(2).permute(2, 0, 1),
#                 memory=memory, memory_pos=memory_pos_embed, num_obj_ptr_tokens=0
#             )
#             pix_feat_with_mem = pix_feat_with_mem.clone().permute(1, 2, 0).view(*pix_feat.shape)
            
#             # ✅ 改动: 传递点提示
#             if point_coords is not None and point_labels is not None:
#                 # 缩放到SAM2输入尺寸
#                 scale_x = self.image_size / original_size[1]
#                 scale_y = self.image_size / original_size[0]
                
#                 scaled_coords = point_coords.copy().astype(np.float32)
#                 scaled_coords[:, 0] *= scale_x
#                 scaled_coords[:, 1] *= scale_y
                
#                 sam_point_coords = torch.from_numpy(scaled_coords).unsqueeze(0).float().to(self.device)
#                 sam_point_labels = torch.from_numpy(point_labels.astype(np.int32)).unsqueeze(0).to(self.device)
#             else:
#                 sam_point_coords = torch.zeros(B, 1, 2, device=self.device)
#                 sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=self.device)
            
#             # 保持原样：mask prompt强度
#             sam_mask_prompt = None
#             if rough_mask is not None:
#                 mask_tensor = self._prepare_mask(rough_mask, target_size=self.image_size)
#                 high_res_mask_logits = mask_tensor.float() * 20.0 - 10.0  # 原始强度
#                 sam_mask_prompt = F.interpolate(high_res_mask_logits, size=(256, 256), mode='bilinear', align_corners=False)
            
#             sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
#                 points=(sam_point_coords, sam_point_labels), boxes=None, masks=sam_mask_prompt
#             )
#             sparse_embeddings = sparse_embeddings.clone()
#             dense_embeddings = dense_embeddings.clone()
#             image_pe = self.model.sam_prompt_encoder.get_dense_pe().clone()
            
#             # 保持原样: multimask=False
#             low_res_masks, ious, _, _ = self.model.sam_mask_decoder(
#                 image_embeddings=pix_feat_with_mem, image_pe=image_pe,
#                 sparse_prompt_embeddings=sparse_embeddings,
#                 dense_prompt_embeddings=dense_embeddings,
#                 multimask_output=False,
#                 repeat_image=False, high_res_features=high_res_features
#             )
            
#             low_res_masks = low_res_masks.float()
#             high_res_masks = F.interpolate(low_res_masks, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
#             final_masks = F.interpolate(high_res_masks, size=original_size, mode='bilinear', align_corners=False)
            
#             pred_mask = (final_masks[0, 0] > 0).cpu().numpy().astype(np.uint8)
#             score = float(ious[0, 0].cpu())
        
#         return pred_mask, score


# # ============================================================
# # 版本C: 不传点提示，只用mask prompt (原始版本，作为对照)
# # ============================================================
# class SAM2MemoryModule_VersionC:
#     """原始版本，完全不改动"""
    
#     def __init__(self, sam2_model, device='cuda', sam2_model_type='large'):
#         self.model = sam2_model
#         self.device = device
#         self.sam2_model_type = sam2_model_type
#         self.prev_out = {}
#         self.support_set = False
#         self.image_size = getattr(sam2_model, 'image_size', 1024)
#         self.num_feature_levels = getattr(sam2_model, 'num_feature_levels', 3)
    
#     def clear_support(self):
#         self.prev_out = {}
#         self.support_set = False
    
#     def _prepare_image(self, img):
#         target_size = self.image_size
#         img_resized = img.resize((target_size, target_size), Image.BILINEAR)
#         img_np = np.array(img_resized).astype(np.float32) / 255.0
#         img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
#         img_tensor = img_tensor.unsqueeze(0).to(self.device)
#         return img_tensor
    
#     def _prepare_mask(self, mask, target_size=None):
#         if target_size is None:
#             target_size = self.image_size
#         mask_resized = cv2.resize(mask.astype(np.float32), (target_size, target_size), interpolation=cv2.INTER_NEAREST)
#         mask_tensor = torch.from_numpy(mask_resized).float()
#         mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
#         return mask_tensor
    
#     def encode_support(self, support_img, support_mask):
#         img_tensor = self._prepare_image(support_img)
        
#         with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
#             backbone_out = self.model.forward_image(img_tensor)
#             feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
#             for i in range(len(feature_maps)):
#                 feature_maps[i] = feature_maps[i].clone()
            
#             pix_feat = feature_maps[-1]
#             mask_tensor = self._prepare_mask(support_mask)
#             high_res_masks = mask_tensor.float() * 20.0 - 10.0
            
#             maskmem_out = self.model.memory_encoder(pix_feat=pix_feat, masks=high_res_masks, skip_mask_sigmoid=True)
            
#             self.prev_out.setdefault("maskmem_features", []).append(maskmem_out["vision_features"].clone())
#             if "maskmem_pos_enc" not in self.prev_out:
#                 self.prev_out["maskmem_pos_enc"] = [m.clone() for m in maskmem_out["vision_pos_enc"]]
        
#         self.support_set = True
    
#     def set_support(self, support_images, support_masks, target_size=None):
#         self.clear_support()
#         for img, mask in zip(support_images, support_masks):
#             self.encode_support(img, mask)
#         return self.support_set
    
#     def predict_query(self, query_img, rough_mask=None, point_coords=None, point_labels=None):
#         if not self.support_set:
#             raise ValueError("请先调用set_support()")
        
#         original_size = query_img.size[::-1]
#         img_tensor = self._prepare_image(query_img)
#         B = img_tensor.shape[0]
        
#         with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
#             backbone_out = self.model.forward_image(img_tensor)
#             feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
#             vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels:]
            
#             for i in range(len(feature_maps)):
#                 feature_maps[i] = feature_maps[i].clone()
#                 vision_pos_embeds[i] = vision_pos_embeds[i].clone()
            
#             high_res_features = feature_maps[:-1]
#             pix_feat = feature_maps[-1]
            
#             to_cat_memory = [m.flatten(2).permute(2, 0, 1) for m in self.prev_out["maskmem_features"]]
#             to_cat_memory_pos_embed = len(self.prev_out["maskmem_features"]) * [
#                 self.prev_out["maskmem_pos_enc"][0].flatten(2).permute(2, 0, 1)
#             ]
            
#             memory = torch.cat(to_cat_memory, dim=0)
#             memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)
            
#             pix_feat_with_mem = self.model.memory_attention(
#                 curr=pix_feat.flatten(2).permute(2, 0, 1),
#                 curr_pos=vision_pos_embeds[-1].flatten(2).permute(2, 0, 1),
#                 memory=memory, memory_pos=memory_pos_embed, num_obj_ptr_tokens=0
#             )
#             pix_feat_with_mem = pix_feat_with_mem.clone().permute(1, 2, 0).view(*pix_feat.shape)
            
#             # 原始：空点
#             sam_point_coords = torch.zeros(B, 1, 2, device=self.device)
#             sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=self.device)
            
#             # 原始：mask强度
#             sam_mask_prompt = None
#             if rough_mask is not None:
#                 mask_tensor = self._prepare_mask(rough_mask, target_size=self.image_size)
#                 high_res_mask_logits = mask_tensor.float() * 20.0 - 10.0
#                 sam_mask_prompt = F.interpolate(high_res_mask_logits, size=(256, 256), mode='bilinear', align_corners=False)
            
#             sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
#                 points=(sam_point_coords, sam_point_labels), boxes=None, masks=sam_mask_prompt
#             )
#             sparse_embeddings = sparse_embeddings.clone()
#             dense_embeddings = dense_embeddings.clone()
#             image_pe = self.model.sam_prompt_encoder.get_dense_pe().clone()
            
#             # 原始：multimask=False
#             low_res_masks, ious, _, _ = self.model.sam_mask_decoder(
#                 image_embeddings=pix_feat_with_mem, image_pe=image_pe,
#                 sparse_prompt_embeddings=sparse_embeddings,
#                 dense_prompt_embeddings=dense_embeddings,
#                 multimask_output=False,
#                 repeat_image=False, high_res_features=high_res_features
#             )
            
#             low_res_masks = low_res_masks.float()
#             high_res_masks = F.interpolate(low_res_masks, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
#             final_masks = F.interpolate(high_res_masks, size=original_size, mode='bilinear', align_corners=False)
            
#             pred_mask = (final_masks[0, 0] > 0).cpu().numpy().astype(np.uint8)
#             score = float(ious[0, 0].cpu())
        
#         return pred_mask, score


# # ============================================================
# # 🔧 选择版本
# # ============================================================
# # 修改这里选择要测试的版本:
# #   'A' = 只改multimask=True
# #   'B' = 传递点提示
# #   'C' = 原始版本(对照)

# USE_VERSION = 'A'  # ⬅️ 修改这里！


# # 根据选择创建Memory模块
# if USE_VERSION == 'A':
#     SAM2MemoryModule = SAM2MemoryModule_VersionA
#     print("📌 使用版本A: 只改 multimask=True")
# elif USE_VERSION == 'B':
#     SAM2MemoryModule = SAM2MemoryModule_VersionB
#     print("📌 使用版本B: 传递点提示")
# else:
#     SAM2MemoryModule = SAM2MemoryModule_VersionC
#     print("📌 使用版本C: 原始版本(对照)")


# # ============================================================
# # SPSAMModel (适配各版本)
# # ============================================================
# class SPSAMModel:
#     def __init__(self, sam2_model, sam2_predictor, dino_model, dino_transform,
#                  device='cuda', sam2_model_type='large'):
#         self.sam2_model = sam2_model
#         self.sam2_predictor = sam2_predictor
#         self.device = device
        
#         self.cmrs = CMRSModule(dino_model, dino_transform, device)
#         self.memory_module = SAM2MemoryModule(sam2_model, device, sam2_model_type)
        
#         self._support_images = []
#         self._support_masks = []
#         self._support_set = False
        
#         print(f"✅ SPSAMModel (版本 {USE_VERSION})")
    
#     def set_support(self, support_images, support_masks):
#         self._support_images = support_images
#         self._support_masks = support_masks
#         success = self.memory_module.set_support(support_images, support_masks)
#         self._support_set = success
#         return success
    
#     def clear_support(self):
#         self._support_images = []
#         self._support_masks = []
#         self._support_set = False
#         self.memory_module.clear_support()
    
#     def predict(self, query_img, support_images, support_masks, use_cmrs=True, use_memory_refinement=False):
#         if use_memory_refinement:
#             self.set_support(support_images, support_masks)
#         return self.predict_query(query_img, use_cmrs, use_memory_refinement, support_images, support_masks)
    
#     def predict_query(self, query_img, use_cmrs=True, use_memory_refinement=False,
#                      support_images=None, support_masks=None):
#         sup_imgs = support_images if support_images else self._support_images
#         sup_masks = support_masks if support_masks else self._support_masks
        
#         if len(sup_imgs) == 0:
#             raise ValueError("No support samples")
        
#         results = {'final_mask': None, 'rough_mask': None, 'similarity_map': None, 'score': 0.0}
        
#         rough_mask = None
#         point_coords = None
#         point_labels = None
        
#         if use_cmrs:
#             rough_mask, similarity_map, point_coords, point_labels = self.cmrs.generate_rough_mask(
#                 query_img, sup_imgs, sup_masks, self.sam2_predictor
#             )
#             results['rough_mask'] = rough_mask
#             results['similarity_map'] = similarity_map
        
#         if use_memory_refinement:
#             if not self._support_set:
#                 self.set_support(sup_imgs, sup_masks)
            
#             pred_mask, score = self.memory_module.predict_query(
#                 query_img, 
#                 rough_mask=rough_mask if use_cmrs else None,
#                 point_coords=point_coords,
#                 point_labels=point_labels
#             )
#             results['final_mask'] = pred_mask
#             results['score'] = score
#         else:
#             if rough_mask is not None:
#                 results['final_mask'] = rough_mask.astype(np.uint8)
#                 results['score'] = 1.0
#             else:
#                 H, W = query_img.size[::-1]
#                 results['final_mask'] = np.zeros((H, W), dtype=np.uint8)
#                 results['score'] = 0.0
        
#         return results


# def visualize_sp_sam_results(query_img, gt_mask, pred_results, support_images=None, support_masks=None, save_path=None, title="SP-SAM Results"):
#     import matplotlib.pyplot as plt
    
#     n_cols = 4
#     n_rows = 1
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4))
    
#     axes[0].imshow(query_img)
#     axes[0].set_title("Query")
#     axes[0].axis('off')
    
#     axes[1].imshow(gt_mask, cmap='gray')
#     axes[1].set_title("GT")
#     axes[1].axis('off')
    
#     rough_mask = pred_results.get('rough_mask')
#     if rough_mask is not None:
#         axes[2].imshow(rough_mask, cmap='gray')
#         axes[2].set_title("Rough")
#     axes[2].axis('off')
    
#     final_mask = pred_results.get('final_mask')
#     if final_mask is not None:
#         axes[3].imshow(final_mask, cmap='gray')
#         if gt_mask is not None:
#             intersection = (final_mask > 0) & (gt_mask > 0)
#             union = (final_mask > 0) | (gt_mask > 0)
#             iou = intersection.sum() / (union.sum() + 1e-8)
#             axes[3].set_title(f"Final (IoU: {iou:.3f})")
#     axes[3].axis('off')
    
#     plt.suptitle(title)
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         plt.close()
#     else:
#         plt.show()


# if __name__ == '__main__':
#     print("=" * 60)
#     print("SP-SAM 渐进式改进测试")
#     print("=" * 60)
#     print(f"\n当前版本: {USE_VERSION}")
#     print("\n修改 USE_VERSION 变量来测试不同版本:")
#     print("  'A' = 只改 multimask=True (推荐先试)")
#     print("  'B' = 传递点提示")
#     print("  'C' = 原始版本(对照)")



# #A+b版本以及消融memory attention的版本
# """
# SP-SAM 版本A+B 消融版
# ======================

# 在版本A+B基础上，可以开关Memory Attention来验证其贡献

# 设置 USE_MEMORY_ATTENTION = False 来禁用Memory Attention
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import cv2
# from PIL import Image
# from typing import List, Dict, Tuple, Optional, Any


# # ============================================================
# # 🔬 消融开关
# # ============================================================
# USE_MEMORY_ATTENTION = False  # ⬅️ 设为False来禁用Memory Attention

# # True  = 完整版本A+B (有Memory Attention)
# # False = 只有multimask + 点提示 + Mask Prompt，无Memory Attention


# class CMRSModule:
#     """CMRS模块"""
    
#     def __init__(self, dino_model, dino_transform, device='cuda'):
#         self.dino_model = dino_model
#         self.dino_transform = dino_transform
#         self.device = device
        
#     def extract_dino_features(self, img_pil):
#         img_tensor = self.dino_transform(img_pil)[None].to(self.device)
#         with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
#             features = self.dino_model.get_intermediate_layers(img_tensor.to(torch.bfloat16))[0]
#             h = w = int(features.shape[1]**0.5)
#             feature_map = features.reshape(1, h, w, -1).permute(0, 3, 1, 2).float()
#         return feature_map.squeeze(0)
    
#     def downsample_mask(self, mask_np, target_size):
#         h, w = target_size
#         resized_mask = cv2.resize(mask_np.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
#         resized_mask = (resized_mask > 0.5).astype(np.float32)
#         return torch.from_numpy(resized_mask).to(self.device)
    
#     def compute_prototype(self, features, mask):
#         mask_sum = mask.sum() + 1e-8
#         prototype = (features * mask.unsqueeze(0)).sum(dim=(1, 2)) / mask_sum
#         return prototype
    
#     def compute_similarity_map_prototype(self, query_features, support_features_list, support_masks_list):
#         C, H, W = query_features.shape
#         prototypes = []
#         for sup_feat, sup_mask in zip(support_features_list, support_masks_list):
#             proto = self.compute_prototype(sup_feat, sup_mask)
#             prototypes.append(proto)
#         prototypes = torch.stack(prototypes, dim=0)
#         avg_prototype = prototypes.mean(dim=0)
#         avg_prototype = F.normalize(avg_prototype, p=2, dim=0)
#         query_flat = query_features.reshape(C, -1).T
#         query_flat = F.normalize(query_flat, p=2, dim=1)
#         similarity_scores = torch.mv(query_flat, avg_prototype)
#         return similarity_scores.reshape(H, W)
    
#     def get_prompts_from_similarity(self, similarity_map, top_k=10, neg_k=5):
#         sim_np = similarity_map.cpu().numpy()
#         H, W = sim_np.shape
#         mean_sim, std_sim, max_sim = sim_np.mean(), sim_np.std(), sim_np.max()
        
#         fg_threshold = max(mean_sim + 1.0 * std_sim, max_sim * 0.5)
#         fg_mask = sim_np > fg_threshold
#         fg_coords = np.argwhere(fg_mask)
        
#         if len(fg_coords) < top_k:
#             fg_threshold = max(mean_sim + 0.5 * std_sim, max_sim * 0.4)
#             fg_coords = np.argwhere(sim_np > fg_threshold)
#         if len(fg_coords) < top_k:
#             fg_coords = np.argwhere(sim_np > mean_sim + 0.3 * std_sim)
        
#         if len(fg_coords) == 0:
#             flat_indices = np.argsort(sim_np.flatten())[-top_k:]
#             pos_points = np.array([np.unravel_index(idx, sim_np.shape) for idx in flat_indices])
#         else:
#             pos_points = self._sample_points_weighted_spatial(fg_coords, sim_np, top_k)
        
#         bg_threshold = mean_sim - 0.5 * std_sim
#         bg_coords = np.argwhere(sim_np < bg_threshold)
        
#         if len(bg_coords) >= neg_k:
#             neg_points = self._sample_negative_points_diverse(bg_coords, sim_np, neg_k, (H, W))
#         else:
#             neg_points = self._sample_from_borders(H, W, neg_k)
        
#         all_points = np.vstack([pos_points, neg_points])
#         all_labels = np.concatenate([np.ones(len(pos_points), dtype=np.int32),
#                                      np.zeros(len(neg_points), dtype=np.int32)])
#         return all_points[:, ::-1], all_labels
    
#     def _sample_points_weighted_spatial(self, coords, sim_np, num_points):
#         sim_values = np.array([sim_np[y, x] for y, x in coords])
#         selected, selected_indices = [], []
#         first_idx = np.argmax(sim_values)
#         selected.append(coords[first_idx])
#         selected_indices.append(first_idx)
        
#         for _ in range(num_points - 1):
#             if len(selected_indices) >= len(coords):
#                 break
#             best_score, best_idx = -np.inf, -1
#             for i, (coord, sim_val) in enumerate(zip(coords, sim_values)):
#                 if i in selected_indices:
#                     continue
#                 min_dist = min(np.sqrt((coord[0]-s[0])**2 + (coord[1]-s[1])**2) for s in selected)
#                 score = sim_val * 0.7 + (min_dist / max(sim_np.shape)) * 0.3
#                 if score > best_score:
#                     best_score, best_idx = score, i
#             if best_idx >= 0:
#                 selected.append(coords[best_idx])
#                 selected_indices.append(best_idx)
#         return np.array(selected)
    
#     def _sample_negative_points_diverse(self, bg_coords, sim_np, num_points, shape):
#         H, W = shape
#         grid_h, grid_w = 3, 3
#         cell_h, cell_w = H // grid_h, W // grid_w
#         neg_points = []
#         for gh in range(grid_h):
#             for gw in range(grid_w):
#                 if len(neg_points) >= num_points:
#                     break
#                 y_start, y_end = gh * cell_h, (gh + 1) * cell_h
#                 x_start, x_end = gw * cell_w, (gw + 1) * cell_w
#                 cell_mask = ((bg_coords[:, 0] >= y_start) & (bg_coords[:, 0] < y_end) &
#                             (bg_coords[:, 1] >= x_start) & (bg_coords[:, 1] < x_end))
#                 cell_coords = bg_coords[cell_mask]
#                 if len(cell_coords) > 0:
#                     cell_sims = np.array([sim_np[y, x] for y, x in cell_coords])
#                     neg_points.append(cell_coords[np.argmin(cell_sims)])
#         if len(neg_points) < num_points and len(bg_coords) > 0:
#             indices = np.random.choice(len(bg_coords), min(num_points - len(neg_points), len(bg_coords)), replace=False)
#             for idx in indices:
#                 neg_points.append(bg_coords[idx])
#         return np.array(neg_points[:num_points])
    
#     def _sample_from_borders(self, H, W, num_points):
#         border = 3
#         corners = [[border, border], [border, W-border-1], [H-border-1, border], [H-border-1, W-border-1]]
#         neg_points = corners[:min(num_points, 4)]
#         while len(neg_points) < num_points:
#             side = np.random.randint(4)
#             if side == 0:
#                 neg_points.append([border, np.random.randint(border, W-border)])
#             elif side == 1:
#                 neg_points.append([H-border-1, np.random.randint(border, W-border)])
#             elif side == 2:
#                 neg_points.append([np.random.randint(border, H-border), border])
#             else:
#                 neg_points.append([np.random.randint(border, H-border), W-border-1])
#         return np.array(neg_points)
    
#     def generate_rough_mask(self, query_img, support_images, support_masks, sam2_predictor, top_k=10, neg_k=5):
#         query_features = self.extract_dino_features(query_img)
#         C, H_feat, W_feat = query_features.shape
        
#         support_features_list, support_masks_list = [], []
#         for sup_img, sup_mask in zip(support_images, support_masks):
#             sup_features = self.extract_dino_features(sup_img)
#             downsampled_mask = self.downsample_mask(sup_mask, (H_feat, W_feat))
#             support_features_list.append(sup_features)
#             support_masks_list.append(downsampled_mask)
        
#         similarity_map = self.compute_similarity_map_prototype(query_features, support_features_list, support_masks_list)
#         point_coords_feat, point_labels = self.get_prompts_from_similarity(similarity_map, top_k, neg_k)
        
#         H_img, W_img = query_img.size[::-1]
#         scale_y, scale_x = H_img / H_feat, W_img / W_feat
        
#         point_coords_img = point_coords_feat.copy().astype(np.float32)
#         point_coords_img[:, 0] *= scale_x
#         point_coords_img[:, 1] *= scale_y
#         point_coords_img = point_coords_img.astype(np.int32)
        
#         sam2_predictor.set_image(np.array(query_img))
        
#         try:
#             masks, scores, _ = sam2_predictor.predict(
#                 point_coords=point_coords_img, point_labels=point_labels, multimask_output=True
#             )
#             rough_mask = masks[np.argmax(scores)]
#         except Exception as e:
#             print(f"   ⚠️  SAM2预测失败: {e}")
#             rough_mask = None
        
#         return rough_mask, similarity_map.cpu().numpy(), point_coords_img, point_labels


# class SAM2MemoryModule:
#     """版本A+B消融版Memory模块"""
    
#     def __init__(self, sam2_model, device='cuda', sam2_model_type='large'):
#         self.model = sam2_model
#         self.device = device
#         self.sam2_model_type = sam2_model_type
#         self.prev_out = {}
#         self.support_set = False
#         self.image_size = getattr(sam2_model, 'image_size', 1024)
#         self.num_feature_levels = getattr(sam2_model, 'num_feature_levels', 3)
        
#         if USE_MEMORY_ATTENTION:
#             print("📌 Memory Attention: ✅ 启用")
#         else:
#             print("📌 Memory Attention: ❌ 禁用 (消融实验)")
    
#     def clear_support(self):
#         self.prev_out = {}
#         self.support_set = False
    
#     def _prepare_image(self, img):
#         target_size = self.image_size
#         img_resized = img.resize((target_size, target_size), Image.BILINEAR)
#         img_np = np.array(img_resized).astype(np.float32) / 255.0
#         img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
#         return img_tensor.unsqueeze(0).to(self.device)
    
#     def _prepare_mask(self, mask, target_size=None):
#         if target_size is None:
#             target_size = self.image_size
#         mask_resized = cv2.resize(mask.astype(np.float32), (target_size, target_size), interpolation=cv2.INTER_NEAREST)
#         mask_tensor = torch.from_numpy(mask_resized).float()
#         return mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
    
#     def encode_support(self, support_img, support_mask):
#         """只有启用Memory Attention时才需要编码support"""
#         if not USE_MEMORY_ATTENTION:
#             self.support_set = True
#             return
        
#         img_tensor = self._prepare_image(support_img)
        
#         with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
#             backbone_out = self.model.forward_image(img_tensor)
#             feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
            
#             for i in range(len(feature_maps)):
#                 feature_maps[i] = feature_maps[i].clone()
            
#             pix_feat = feature_maps[-1]
#             mask_tensor = self._prepare_mask(support_mask)
#             high_res_masks = mask_tensor.float() * 20.0 - 10.0
            
#             maskmem_out = self.model.memory_encoder(pix_feat=pix_feat, masks=high_res_masks, skip_mask_sigmoid=True)
            
#             self.prev_out.setdefault("maskmem_features", []).append(maskmem_out["vision_features"].clone())
#             if "maskmem_pos_enc" not in self.prev_out:
#                 self.prev_out["maskmem_pos_enc"] = [m.clone() for m in maskmem_out["vision_pos_enc"]]
        
#         self.support_set = True
    
#     def set_support(self, support_images, support_masks, target_size=None):
#         self.clear_support()
#         for img, mask in zip(support_images, support_masks):
#             self.encode_support(img, mask)
#         return self.support_set
    
#     def predict_query(self, query_img, rough_mask=None, point_coords=None, point_labels=None):
#         if not self.support_set:
#             raise ValueError("请先调用set_support()")
        
#         original_size = query_img.size[::-1]
#         img_tensor = self._prepare_image(query_img)
#         B = img_tensor.shape[0]
        
#         with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
#             backbone_out = self.model.forward_image(img_tensor)
#             feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
#             vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels:]
            
#             for i in range(len(feature_maps)):
#                 feature_maps[i] = feature_maps[i].clone()
#                 vision_pos_embeds[i] = vision_pos_embeds[i].clone()
            
#             high_res_features = feature_maps[:-1]
#             pix_feat = feature_maps[-1]
            
#             # ============================================================
#             # 🔬 消融点: Memory Attention
#             # ============================================================
#             if USE_MEMORY_ATTENTION:
#                 # 使用Memory Attention
#                 to_cat_memory = [m.flatten(2).permute(2, 0, 1) for m in self.prev_out["maskmem_features"]]
#                 to_cat_memory_pos_embed = len(self.prev_out["maskmem_features"]) * [
#                     self.prev_out["maskmem_pos_enc"][0].flatten(2).permute(2, 0, 1)
#                 ]
                
#                 memory = torch.cat(to_cat_memory, dim=0)
#                 memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)
                
#                 pix_feat_with_mem = self.model.memory_attention(
#                     curr=pix_feat.flatten(2).permute(2, 0, 1),
#                     curr_pos=vision_pos_embeds[-1].flatten(2).permute(2, 0, 1),
#                     memory=memory, memory_pos=memory_pos_embed, num_obj_ptr_tokens=0
#                 )
#                 pix_feat_with_mem = pix_feat_with_mem.clone().permute(1, 2, 0).view(*pix_feat.shape)
#             else:
#                 # ❌ 禁用Memory Attention，直接使用原始特征
#                 pix_feat_with_mem = pix_feat
            
#             # 点提示 (版本B)
#             if point_coords is not None and point_labels is not None:
#                 scale_x = self.image_size / original_size[1]
#                 scale_y = self.image_size / original_size[0]
                
#                 scaled_coords = point_coords.copy().astype(np.float32)
#                 scaled_coords[:, 0] *= scale_x
#                 scaled_coords[:, 1] *= scale_y
                
#                 sam_point_coords = torch.from_numpy(scaled_coords).unsqueeze(0).float().to(self.device)
#                 sam_point_labels = torch.from_numpy(point_labels.astype(np.int32)).unsqueeze(0).to(self.device)
#             else:
#                 sam_point_coords = torch.zeros(B, 1, 2, device=self.device)
#                 sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=self.device)
            
#             # Mask Prompt
#             sam_mask_prompt = None
#             if rough_mask is not None:
#                 mask_tensor = self._prepare_mask(rough_mask, target_size=self.image_size)
#                 high_res_mask_logits = mask_tensor.float() * 20.0 - 10.0
#                 sam_mask_prompt = F.interpolate(high_res_mask_logits, size=(256, 256),
#                                                 mode='bilinear', align_corners=False)
            
#             sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
#                 points=(sam_point_coords, sam_point_labels), boxes=None, masks=sam_mask_prompt
#             )
#             sparse_embeddings = sparse_embeddings.clone()
#             dense_embeddings = dense_embeddings.clone()
#             image_pe = self.model.sam_prompt_encoder.get_dense_pe().clone()
            
#             # multimask=True (版本A)
#             low_res_masks, ious, _, _ = self.model.sam_mask_decoder(
#                 image_embeddings=pix_feat_with_mem, image_pe=image_pe,
#                 sparse_prompt_embeddings=sparse_embeddings,
#                 dense_prompt_embeddings=dense_embeddings,
#                 multimask_output=True,
#                 repeat_image=False, high_res_features=high_res_features
#             )
            
#             best_idx = torch.argmax(ious[0])
#             best_mask = low_res_masks[0, best_idx:best_idx+1].unsqueeze(0)
#             best_iou = ious[0, best_idx]
            
#             best_mask = best_mask.float()
#             high_res_masks = F.interpolate(best_mask, size=(self.image_size, self.image_size),
#                                            mode='bilinear', align_corners=False)
#             final_masks = F.interpolate(high_res_masks, size=original_size,
#                                         mode='bilinear', align_corners=False)
            
#             pred_mask = (final_masks[0, 0] > 0).cpu().numpy().astype(np.uint8)
#             score = float(best_iou.cpu())
        
#         return pred_mask, score


# class SPSAMModel:
#     def __init__(self, sam2_model, sam2_predictor, dino_model, dino_transform,
#                  device='cuda', sam2_model_type='large'):
#         self.sam2_model = sam2_model
#         self.sam2_predictor = sam2_predictor
#         self.device = device
        
#         self.cmrs = CMRSModule(dino_model, dino_transform, device)
#         self.memory_module = SAM2MemoryModule(sam2_model, device, sam2_model_type)
        
#         self._support_images = []
#         self._support_masks = []
#         self._support_set = False
        
#         mode = "完整版A+B" if USE_MEMORY_ATTENTION else "消融版(无Memory Attention)"
#         print(f"✅ SPSAMModel ({mode})")
    
#     def set_support(self, support_images, support_masks):
#         self._support_images = support_images
#         self._support_masks = support_masks
#         success = self.memory_module.set_support(support_images, support_masks)
#         self._support_set = success
#         return success
    
#     def clear_support(self):
#         self._support_images = []
#         self._support_masks = []
#         self._support_set = False
#         self.memory_module.clear_support()
    
#     def predict(self, query_img, support_images, support_masks, use_cmrs=True, use_memory_refinement=False):
#         if use_memory_refinement:
#             self.set_support(support_images, support_masks)
#         return self.predict_query(query_img, use_cmrs, use_memory_refinement, support_images, support_masks)
    
#     def predict_query(self, query_img, use_cmrs=True, use_memory_refinement=False,
#                      support_images=None, support_masks=None):
#         sup_imgs = support_images if support_images else self._support_images
#         sup_masks = support_masks if support_masks else self._support_masks
        
#         if len(sup_imgs) == 0:
#             raise ValueError("No support samples")
        
#         results = {'final_mask': None, 'rough_mask': None, 'similarity_map': None, 'score': 0.0}
        
#         rough_mask = None
#         point_coords = None
#         point_labels = None
        
#         if use_cmrs:
#             rough_mask, similarity_map, point_coords, point_labels = self.cmrs.generate_rough_mask(
#                 query_img, sup_imgs, sup_masks, self.sam2_predictor
#             )
#             results['rough_mask'] = rough_mask
#             results['similarity_map'] = similarity_map
        
#         if use_memory_refinement:
#             if not self._support_set:
#                 self.set_support(sup_imgs, sup_masks)
            
#             pred_mask, score = self.memory_module.predict_query(
#                 query_img, 
#                 rough_mask=rough_mask if use_cmrs else None,
#                 point_coords=point_coords,
#                 point_labels=point_labels
#             )
#             results['final_mask'] = pred_mask
#             results['score'] = score
#         else:
#             if rough_mask is not None:
#                 results['final_mask'] = rough_mask.astype(np.uint8)
#                 results['score'] = 1.0
#             else:
#                 H, W = query_img.size[::-1]
#                 results['final_mask'] = np.zeros((H, W), dtype=np.uint8)
#                 results['score'] = 0.0
        
#         return results


# def visualize_sp_sam_results(query_img, gt_mask, pred_results,
#                             support_images=None, support_masks=None,
#                             save_path=None, title="SP-SAM Results"):
#     import matplotlib.pyplot as plt
#     n_cols = 4
#     fig, axes = plt.subplots(1, n_cols, figsize=(16, 4))
    
#     axes[0].imshow(query_img)
#     axes[0].set_title("Query")
#     axes[0].axis('off')
    
#     axes[1].imshow(gt_mask, cmap='gray')
#     axes[1].set_title("GT")
#     axes[1].axis('off')
    
#     if pred_results.get('rough_mask') is not None:
#         axes[2].imshow(pred_results['rough_mask'], cmap='gray')
#         axes[2].set_title("Rough")
#     axes[2].axis('off')
    
#     if pred_results.get('final_mask') is not None:
#         axes[3].imshow(pred_results['final_mask'], cmap='gray')
#         if gt_mask is not None:
#             intersection = (pred_results['final_mask'] > 0) & (gt_mask > 0)
#             union = (pred_results['final_mask'] > 0) | (gt_mask > 0)
#             iou = intersection.sum() / (union.sum() + 1e-8)
#             axes[3].set_title(f"Final (IoU: {iou:.3f})")
#     axes[3].axis('off')
    
#     plt.suptitle(title)
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         plt.close()
#     else:
#         plt.show()


# if __name__ == '__main__':
#     print("=" * 60)
#     print("SP-SAM 版本A+B 消融实验")
#     print("=" * 60)
#     print(f"\nUSE_MEMORY_ATTENTION = {USE_MEMORY_ATTENTION}")
#     print("\n设置说明:")
#     print("  True  = 完整版本A+B (有Memory Attention)")
#     print("  False = 消融版 (只有multimask + 点提示 + Mask Prompt)")






# """
# SP-SAM CMRS改进版
# ==================

# 真正的瓶颈在CMRS (Rough Mask只有60.88%)
# Memory只能在此基础上修补，无法突破上限

# 改进方向：
# 1. 多尺度DINO特征融合
# 2. 更好的Prototype计算（前景/背景双原型）
# 3. 更智能的点采样策略
# 4. Box Prompt辅助
# 5. 多轮CMRS迭代

# 选择 CMRS_MODE 来测试
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import cv2
# from PIL import Image
# from typing import List, Dict, Tuple, Optional, Any


# # ============================================================
# # 🔧 选择CMRS改进方案
# # ============================================================
# CMRS_MODE = 'dense_points'  # ⬅️ 修改这里

# # 可选值:
# # 'baseline'        - 原始CMRS (60.88%)
# # 'dual_prototype'  - 前景+背景双原型对比 整体63.96%MIOU
# # 'multi_scale'     - 多尺度DINO特征 整体64.97%MIOU
# # 'box_prompt'      - 添加Box Prompt 整体62.39%MIOU
# # 'iterative_cmrs'  - 迭代CMRS (用rough mask更新prototype) 整体62.30%MIOU
# # 'dense_points'    - 更多的点提示  整体63.66%MIOU


# class CMRSModule:
#     """改进版CMRS模块"""
    
#     def __init__(self, dino_model, dino_transform, device='cuda'):
#         self.dino_model = dino_model
#         self.dino_transform = dino_transform
#         self.device = device
#         self.current_prototype = None
#         self.current_bg_prototype = None
#         self.current_query_features = None
        
#         print(f"📌 CMRS模式: {CMRS_MODE}")
        
#     def extract_dino_features(self, img_pil):
#         img_tensor = self.dino_transform(img_pil)[None].to(self.device)
#         with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
#             features = self.dino_model.get_intermediate_layers(img_tensor.to(torch.bfloat16))[0]
#             h = w = int(features.shape[1]**0.5)
#             feature_map = features.reshape(1, h, w, -1).permute(0, 3, 1, 2).float()
#         return feature_map.squeeze(0)
    
#     def extract_dino_features_multiscale(self, img_pil):
#         """多尺度特征提取"""
#         scales = [0.75, 1.0, 1.25]
#         original_size = img_pil.size
        
#         all_features = []
#         for scale in scales:
#             new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
#             img_scaled = img_pil.resize(new_size, Image.BILINEAR)
            
#             img_tensor = self.dino_transform(img_scaled)[None].to(self.device)
#             with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
#                 features = self.dino_model.get_intermediate_layers(img_tensor.to(torch.bfloat16))[0]
#                 h = w = int(features.shape[1]**0.5)
#                 feature_map = features.reshape(1, h, w, -1).permute(0, 3, 1, 2).float()
            
#             all_features.append(feature_map.squeeze(0))
        
#         # 统一到中间尺度的大小
#         target_h, target_w = all_features[1].shape[1], all_features[1].shape[2]
#         fused_features = []
#         for feat in all_features:
#             if feat.shape[1] != target_h or feat.shape[2] != target_w:
#                 feat = F.interpolate(feat.unsqueeze(0), size=(target_h, target_w), 
#                                     mode='bilinear', align_corners=False).squeeze(0)
#             fused_features.append(feat)
        
#         # 平均融合
#         fused = torch.stack(fused_features, dim=0).mean(dim=0)
#         return fused
    
#     def downsample_mask(self, mask_np, target_size):
#         h, w = target_size
#         resized_mask = cv2.resize(mask_np.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
#         resized_mask = (resized_mask > 0.5).astype(np.float32)
#         return torch.from_numpy(resized_mask).to(self.device)
    
#     def compute_prototype(self, features, mask):
#         """计算前景原型"""
#         mask_sum = mask.sum() + 1e-8
#         prototype = (features * mask.unsqueeze(0)).sum(dim=(1, 2)) / mask_sum
#         return prototype
    
#     def compute_bg_prototype(self, features, mask):
#         """计算背景原型"""
#         bg_mask = 1.0 - mask
#         bg_sum = bg_mask.sum() + 1e-8
#         bg_prototype = (features * bg_mask.unsqueeze(0)).sum(dim=(1, 2)) / bg_sum
#         return bg_prototype
    
#     def compute_similarity_map_dual_prototype(self, query_features, fg_prototype, bg_prototype):
#         """双原型对比：前景相似度 - 背景相似度"""
#         C, H, W = query_features.shape
        
#         fg_proto = F.normalize(fg_prototype, p=2, dim=0)
#         bg_proto = F.normalize(bg_prototype, p=2, dim=0)
        
#         query_flat = query_features.reshape(C, -1).T
#         query_flat = F.normalize(query_flat, p=2, dim=1)
        
#         fg_sim = torch.mv(query_flat, fg_proto)
#         bg_sim = torch.mv(query_flat, bg_proto)
        
#         # 对比分数：更像前景 vs 更像背景
#         contrast_score = fg_sim - bg_sim
        
#         return contrast_score.reshape(H, W)
    
#     def compute_similarity_map_prototype(self, query_features, support_features_list, support_masks_list):
#         C, H, W = query_features.shape
        
#         fg_prototypes = []
#         bg_prototypes = []
        
#         for sup_feat, sup_mask in zip(support_features_list, support_masks_list):
#             fg_proto = self.compute_prototype(sup_feat, sup_mask)
#             fg_prototypes.append(fg_proto)
            
#             if CMRS_MODE == 'dual_prototype':
#                 bg_proto = self.compute_bg_prototype(sup_feat, sup_mask)
#                 bg_prototypes.append(bg_proto)
        
#         avg_fg_prototype = torch.stack(fg_prototypes, dim=0).mean(dim=0)
#         self.current_prototype = F.normalize(avg_fg_prototype, p=2, dim=0)
        
#         if CMRS_MODE == 'dual_prototype' and bg_prototypes:
#             avg_bg_prototype = torch.stack(bg_prototypes, dim=0).mean(dim=0)
#             self.current_bg_prototype = F.normalize(avg_bg_prototype, p=2, dim=0)
            
#             # 使用双原型对比
#             similarity_map = self.compute_similarity_map_dual_prototype(
#                 query_features, self.current_prototype, self.current_bg_prototype
#             )
#         else:
#             # 原始单原型
#             query_flat = query_features.reshape(C, -1).T
#             query_flat = F.normalize(query_flat, p=2, dim=1)
#             similarity_scores = torch.mv(query_flat, self.current_prototype)
#             similarity_map = similarity_scores.reshape(H, W)
        
#         self.current_query_features = query_features
#         return similarity_map
    
#     def get_prompts_from_similarity(self, similarity_map, top_k=10, neg_k=5):
#         sim_np = similarity_map.cpu().numpy()
#         H, W = sim_np.shape
#         mean_sim, std_sim, max_sim = sim_np.mean(), sim_np.std(), sim_np.max()
#         min_sim = sim_np.min()
        
#         # 根据模式调整阈值
#         if CMRS_MODE == 'dual_prototype':
#             # 双原型模式：分数在[-1, 1]之间，>0表示更像前景
#             fg_threshold = max(0.1, mean_sim + 0.5 * std_sim)
#             bg_threshold = min(-0.1, mean_sim - 0.5 * std_sim)
#         else:
#             fg_threshold = max(mean_sim + 1.0 * std_sim, max_sim * 0.5)
#             bg_threshold = mean_sim - 0.5 * std_sim
        
#         fg_coords = np.argwhere(sim_np > fg_threshold)
        
#         if len(fg_coords) < top_k:
#             fg_threshold = mean_sim + 0.3 * std_sim if CMRS_MODE != 'dual_prototype' else 0.0
#             fg_coords = np.argwhere(sim_np > fg_threshold)
        
#         if len(fg_coords) == 0:
#             flat_indices = np.argsort(sim_np.flatten())[-top_k:]
#             pos_points = np.array([np.unravel_index(idx, sim_np.shape) for idx in flat_indices])
#         else:
#             pos_points = self._sample_points_weighted_spatial(fg_coords, sim_np, top_k)
        
#         bg_coords = np.argwhere(sim_np < bg_threshold)
        
#         if len(bg_coords) >= neg_k:
#             neg_points = self._sample_negative_points_diverse(bg_coords, sim_np, neg_k, (H, W))
#         else:
#             neg_points = self._sample_from_borders(H, W, neg_k)
        
#         # Dense points模式：增加点数量
#         if CMRS_MODE == 'dense_points':
#             # 额外采样更多正负点
#             extra_pos = self._sample_points_weighted_spatial(fg_coords, sim_np, 10) if len(fg_coords) > 10 else pos_points
#             extra_neg = self._sample_negative_points_diverse(bg_coords, sim_np, 10, (H, W)) if len(bg_coords) > 10 else neg_points
#             pos_points = np.vstack([pos_points, extra_pos])
#             neg_points = np.vstack([neg_points, extra_neg])
#             # 去重
#             pos_points = np.unique(pos_points, axis=0)[:20]
#             neg_points = np.unique(neg_points, axis=0)[:10]
        
#         all_points = np.vstack([pos_points, neg_points])
#         all_labels = np.concatenate([np.ones(len(pos_points), dtype=np.int32),
#                                      np.zeros(len(neg_points), dtype=np.int32)])
#         return all_points[:, ::-1], all_labels
    
#     def get_box_from_similarity(self, similarity_map, threshold_ratio=0.5):
#         """从相似度图生成Box Prompt"""
#         sim_np = similarity_map.cpu().numpy()
        
#         if CMRS_MODE == 'dual_prototype':
#             threshold = 0.0  # >0 表示前景
#         else:
#             threshold = sim_np.mean() + 0.5 * sim_np.std()
        
#         fg_mask = (sim_np > threshold).astype(np.uint8)
        
#         # 找轮廓
#         contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         if not contours:
#             return None
        
#         # 找最大轮廓的边界框
#         largest_contour = max(contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(largest_contour)
        
#         # 稍微扩大一点
#         margin = 2
#         x = max(0, x - margin)
#         y = max(0, y - margin)
#         w = min(sim_np.shape[1] - x, w + 2 * margin)
#         h = min(sim_np.shape[0] - y, h + 2 * margin)
        
#         return np.array([x, y, x + w, y + h])
    
#     def _sample_points_weighted_spatial(self, coords, sim_np, num_points):
#         if len(coords) == 0:
#             return np.array([[sim_np.shape[0]//2, sim_np.shape[1]//2]])
        
#         sim_values = np.array([sim_np[y, x] for y, x in coords])
#         selected, selected_indices = [], []
#         first_idx = np.argmax(sim_values)
#         selected.append(coords[first_idx])
#         selected_indices.append(first_idx)
        
#         for _ in range(num_points - 1):
#             if len(selected_indices) >= len(coords):
#                 break
#             best_score, best_idx = -np.inf, -1
#             for i, (coord, sim_val) in enumerate(zip(coords, sim_values)):
#                 if i in selected_indices:
#                     continue
#                 min_dist = min(np.sqrt((coord[0]-s[0])**2 + (coord[1]-s[1])**2) for s in selected)
#                 score = sim_val * 0.7 + (min_dist / max(sim_np.shape)) * 0.3
#                 if score > best_score:
#                     best_score, best_idx = score, i
#             if best_idx >= 0:
#                 selected.append(coords[best_idx])
#                 selected_indices.append(best_idx)
#         return np.array(selected)
    
#     def _sample_negative_points_diverse(self, bg_coords, sim_np, num_points, shape):
#         H, W = shape
#         grid_h, grid_w = 3, 3
#         cell_h, cell_w = H // grid_h, W // grid_w
#         neg_points = []
#         for gh in range(grid_h):
#             for gw in range(grid_w):
#                 if len(neg_points) >= num_points:
#                     break
#                 y_start, y_end = gh * cell_h, (gh + 1) * cell_h
#                 x_start, x_end = gw * cell_w, (gw + 1) * cell_w
#                 cell_mask = ((bg_coords[:, 0] >= y_start) & (bg_coords[:, 0] < y_end) &
#                             (bg_coords[:, 1] >= x_start) & (bg_coords[:, 1] < x_end))
#                 cell_coords = bg_coords[cell_mask]
#                 if len(cell_coords) > 0:
#                     cell_sims = np.array([sim_np[y, x] for y, x in cell_coords])
#                     neg_points.append(cell_coords[np.argmin(cell_sims)])
#         if len(neg_points) < num_points and len(bg_coords) > 0:
#             indices = np.random.choice(len(bg_coords), min(num_points - len(neg_points), len(bg_coords)), replace=False)
#             for idx in indices:
#                 neg_points.append(bg_coords[idx])
#         return np.array(neg_points[:num_points]) if neg_points else np.array([[0, 0]])
    
#     def _sample_from_borders(self, H, W, num_points):
#         border = 3
#         corners = [[border, border], [border, W-border-1], [H-border-1, border], [H-border-1, W-border-1]]
#         neg_points = corners[:min(num_points, 4)]
#         while len(neg_points) < num_points:
#             side = np.random.randint(4)
#             if side == 0:
#                 neg_points.append([border, np.random.randint(border, W-border)])
#             elif side == 1:
#                 neg_points.append([H-border-1, np.random.randint(border, W-border)])
#             elif side == 2:
#                 neg_points.append([np.random.randint(border, H-border), border])
#             else:
#                 neg_points.append([np.random.randint(border, H-border), W-border-1])
#         return np.array(neg_points)
    
#     def generate_rough_mask(self, query_img, support_images, support_masks, sam2_predictor, top_k=10, neg_k=5):
#         # 特征提取
#         if CMRS_MODE == 'multi_scale':
#             query_features = self.extract_dino_features_multiscale(query_img)
#         else:
#             query_features = self.extract_dino_features(query_img)
        
#         C, H_feat, W_feat = query_features.shape
        
#         support_features_list, support_masks_list = [], []
#         for sup_img, sup_mask in zip(support_images, support_masks):
#             if CMRS_MODE == 'multi_scale':
#                 sup_features = self.extract_dino_features_multiscale(sup_img)
#             else:
#                 sup_features = self.extract_dino_features(sup_img)
            
#             # 确保mask和特征尺寸匹配
#             downsampled_mask = self.downsample_mask(sup_mask, (sup_features.shape[1], sup_features.shape[2]))
#             support_features_list.append(sup_features)
#             support_masks_list.append(downsampled_mask)
        
#         similarity_map = self.compute_similarity_map_prototype(query_features, support_features_list, support_masks_list)
#         point_coords_feat, point_labels = self.get_prompts_from_similarity(similarity_map, top_k, neg_k)
        
#         H_img, W_img = query_img.size[::-1]
#         H_feat, W_feat = similarity_map.shape
#         scale_y, scale_x = H_img / H_feat, W_img / W_feat
        
#         point_coords_img = point_coords_feat.copy().astype(np.float32)
#         point_coords_img[:, 0] *= scale_x
#         point_coords_img[:, 1] *= scale_y
#         point_coords_img = point_coords_img.astype(np.int32)
        
#         sam2_predictor.set_image(np.array(query_img))
        
#         # Box Prompt
#         box_prompt = None
#         if CMRS_MODE == 'box_prompt':
#             box_feat = self.get_box_from_similarity(similarity_map)
#             if box_feat is not None:
#                 box_prompt = box_feat.astype(np.float32)
#                 box_prompt[0] *= scale_x
#                 box_prompt[1] *= scale_y
#                 box_prompt[2] *= scale_x
#                 box_prompt[3] *= scale_y
#                 box_prompt = box_prompt.astype(np.int32)
        
#         try:
#             if box_prompt is not None:
#                 masks, scores, _ = sam2_predictor.predict(
#                     point_coords=point_coords_img, 
#                     point_labels=point_labels,
#                     box=box_prompt,
#                     multimask_output=True
#                 )
#             else:
#                 masks, scores, _ = sam2_predictor.predict(
#                     point_coords=point_coords_img, 
#                     point_labels=point_labels, 
#                     multimask_output=True
#                 )
#             rough_mask = masks[np.argmax(scores)]
#         except Exception as e:
#             print(f"   ⚠️  SAM2预测失败: {e}")
#             rough_mask = None
        
#         # 迭代CMRS
#         if CMRS_MODE == 'iterative_cmrs' and rough_mask is not None:
#             # 用rough_mask更新prototype，再预测一次
#             rough_mask_downsampled = self.downsample_mask(rough_mask, (H_feat, W_feat))
            
#             # 更新query的prototype
#             refined_prototype = self.compute_prototype(query_features, rough_mask_downsampled)
#             combined_prototype = F.normalize(
#                 0.7 * self.current_prototype + 0.3 * refined_prototype, p=2, dim=0
#             )
            
#             # 重新计算相似度
#             query_flat = query_features.reshape(C, -1).T
#             query_flat = F.normalize(query_flat, p=2, dim=1)
#             refined_sim = torch.mv(query_flat, combined_prototype).reshape(H_feat, W_feat)
            
#             # 重新采样点
#             point_coords_feat2, point_labels2 = self.get_prompts_from_similarity(refined_sim, top_k, neg_k)
#             point_coords_img2 = point_coords_feat2.copy().astype(np.float32)
#             point_coords_img2[:, 0] *= scale_x
#             point_coords_img2[:, 1] *= scale_y
#             point_coords_img2 = point_coords_img2.astype(np.int32)
            
#             try:
#                 masks2, scores2, _ = sam2_predictor.predict(
#                     point_coords=point_coords_img2, 
#                     point_labels=point_labels2, 
#                     multimask_output=True
#                 )
#                 rough_mask = masks2[np.argmax(scores2)]
#                 similarity_map = refined_sim
#                 point_coords_img = point_coords_img2
#             except:
#                 pass  # 保持原来的结果
        
#         return rough_mask, similarity_map.cpu().numpy(), point_coords_img, point_labels


# class SAM2MemoryModule:
#     """Memory模块 - 保持A+B版本"""
    
#     def __init__(self, sam2_model, device='cuda', sam2_model_type='large'):
#         self.model = sam2_model
#         self.device = device
#         self.sam2_model_type = sam2_model_type
#         self.prev_out = {}
#         self.support_set = False
#         self.image_size = getattr(sam2_model, 'image_size', 1024)
#         self.num_feature_levels = getattr(sam2_model, 'num_feature_levels', 3)
    
#     def clear_support(self):
#         self.prev_out = {}
#         self.support_set = False
    
#     def _prepare_image(self, img):
#         target_size = self.image_size
#         img_resized = img.resize((target_size, target_size), Image.BILINEAR)
#         img_np = np.array(img_resized).astype(np.float32) / 255.0
#         img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
#         return img_tensor.unsqueeze(0).to(self.device)
    
#     def _prepare_mask(self, mask, target_size=None):
#         if target_size is None:
#             target_size = self.image_size
#         mask_resized = cv2.resize(mask.astype(np.float32), (target_size, target_size), interpolation=cv2.INTER_NEAREST)
#         mask_tensor = torch.from_numpy(mask_resized).float()
#         return mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
    
#     def encode_support(self, support_img, support_mask):
#         img_tensor = self._prepare_image(support_img)
        
#         with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
#             backbone_out = self.model.forward_image(img_tensor)
#             feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
            
#             for i in range(len(feature_maps)):
#                 feature_maps[i] = feature_maps[i].clone()
            
#             pix_feat = feature_maps[-1]
#             mask_tensor = self._prepare_mask(support_mask)
#             high_res_masks = mask_tensor.float() * 20.0 - 10.0
            
#             maskmem_out = self.model.memory_encoder(pix_feat=pix_feat, masks=high_res_masks, skip_mask_sigmoid=True)
            
#             self.prev_out.setdefault("maskmem_features", []).append(maskmem_out["vision_features"].clone())
#             if "maskmem_pos_enc" not in self.prev_out:
#                 self.prev_out["maskmem_pos_enc"] = [m.clone() for m in maskmem_out["vision_pos_enc"]]
        
#         self.support_set = True
    
#     def set_support(self, support_images, support_masks, target_size=None):
#         self.clear_support()
#         for img, mask in zip(support_images, support_masks):
#             self.encode_support(img, mask)
#         return self.support_set
    
#     def predict_query(self, query_img, rough_mask=None, point_coords=None, point_labels=None):
#         if not self.support_set:
#             raise ValueError("请先调用set_support()")
        
#         original_size = query_img.size[::-1]
#         img_tensor = self._prepare_image(query_img)
#         B = img_tensor.shape[0]
        
#         with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
#             backbone_out = self.model.forward_image(img_tensor)
#             feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
#             vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels:]
            
#             for i in range(len(feature_maps)):
#                 feature_maps[i] = feature_maps[i].clone()
#                 vision_pos_embeds[i] = vision_pos_embeds[i].clone()
            
#             high_res_features = feature_maps[:-1]
#             pix_feat = feature_maps[-1]
            
#             # Memory Attention
#             to_cat_memory = [m.flatten(2).permute(2, 0, 1) for m in self.prev_out["maskmem_features"]]
#             to_cat_memory_pos_embed = len(self.prev_out["maskmem_features"]) * [
#                 self.prev_out["maskmem_pos_enc"][0].flatten(2).permute(2, 0, 1)
#             ]
            
#             memory = torch.cat(to_cat_memory, dim=0)
#             memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)
            
#             pix_feat_with_mem = self.model.memory_attention(
#                 curr=pix_feat.flatten(2).permute(2, 0, 1),
#                 curr_pos=vision_pos_embeds[-1].flatten(2).permute(2, 0, 1),
#                 memory=memory, memory_pos=memory_pos_embed, num_obj_ptr_tokens=0
#             )
#             pix_feat_with_mem = pix_feat_with_mem.clone().permute(1, 2, 0).view(*pix_feat.shape)
            
#             # 点提示
#             if point_coords is not None and point_labels is not None:
#                 scale_x = self.image_size / original_size[1]
#                 scale_y = self.image_size / original_size[0]
                
#                 scaled_coords = point_coords.copy().astype(np.float32)
#                 scaled_coords[:, 0] *= scale_x
#                 scaled_coords[:, 1] *= scale_y
                
#                 sam_point_coords = torch.from_numpy(scaled_coords).unsqueeze(0).float().to(self.device)
#                 sam_point_labels = torch.from_numpy(point_labels.astype(np.int32)).unsqueeze(0).to(self.device)
#             else:
#                 sam_point_coords = torch.zeros(B, 1, 2, device=self.device)
#                 sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=self.device)
            
#             # Mask Prompt
#             sam_mask_prompt = None
#             if rough_mask is not None:
#                 mask_tensor = self._prepare_mask(rough_mask, target_size=self.image_size)
#                 high_res_mask_logits = mask_tensor.float() * 20.0 - 10.0
#                 sam_mask_prompt = F.interpolate(high_res_mask_logits, size=(256, 256),
#                                                 mode='bilinear', align_corners=False)
            
#             sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
#                 points=(sam_point_coords, sam_point_labels), boxes=None, masks=sam_mask_prompt
#             )
#             sparse_embeddings = sparse_embeddings.clone()
#             dense_embeddings = dense_embeddings.clone()
#             image_pe = self.model.sam_prompt_encoder.get_dense_pe().clone()
            
#             low_res_masks, ious, _, _ = self.model.sam_mask_decoder(
#                 image_embeddings=pix_feat_with_mem, image_pe=image_pe,
#                 sparse_prompt_embeddings=sparse_embeddings,
#                 dense_prompt_embeddings=dense_embeddings,
#                 multimask_output=True,
#                 repeat_image=False, high_res_features=high_res_features
#             )
            
#             best_idx = torch.argmax(ious[0])
#             best_mask = low_res_masks[0, best_idx:best_idx+1].unsqueeze(0)
#             best_iou = ious[0, best_idx]
            
#             best_mask = best_mask.float()
#             high_res_masks = F.interpolate(best_mask, size=(self.image_size, self.image_size),
#                                            mode='bilinear', align_corners=False)
#             final_masks = F.interpolate(high_res_masks, size=original_size,
#                                         mode='bilinear', align_corners=False)
            
#             pred_mask = (final_masks[0, 0] > 0).cpu().numpy().astype(np.uint8)
#             score = float(best_iou.cpu())
        
#         return pred_mask, score


# class SPSAMModel:
#     def __init__(self, sam2_model, sam2_predictor, dino_model, dino_transform,
#                  device='cuda', sam2_model_type='large'):
#         self.sam2_model = sam2_model
#         self.sam2_predictor = sam2_predictor
#         self.device = device
        
#         self.cmrs = CMRSModule(dino_model, dino_transform, device)
#         self.memory_module = SAM2MemoryModule(sam2_model, device, sam2_model_type)
        
#         self._support_images = []
#         self._support_masks = []
#         self._support_set = False
        
#         print(f"✅ SPSAMModel (CMRS改进: {CMRS_MODE})")
    
#     def set_support(self, support_images, support_masks):
#         self._support_images = support_images
#         self._support_masks = support_masks
#         success = self.memory_module.set_support(support_images, support_masks)
#         self._support_set = success
#         return success
    
#     def clear_support(self):
#         self._support_images = []
#         self._support_masks = []
#         self._support_set = False
#         self.memory_module.clear_support()
    
#     def predict(self, query_img, support_images, support_masks, use_cmrs=True, use_memory_refinement=False):
#         if use_memory_refinement:
#             self.set_support(support_images, support_masks)
#         return self.predict_query(query_img, use_cmrs, use_memory_refinement, support_images, support_masks)
    
#     def predict_query(self, query_img, use_cmrs=True, use_memory_refinement=False,
#                      support_images=None, support_masks=None):
#         sup_imgs = support_images if support_images else self._support_images
#         sup_masks = support_masks if support_masks else self._support_masks
        
#         if len(sup_imgs) == 0:
#             raise ValueError("No support samples")
        
#         results = {'final_mask': None, 'rough_mask': None, 'similarity_map': None, 'score': 0.0}
        
#         rough_mask = None
#         point_coords = None
#         point_labels = None
        
#         if use_cmrs:
#             rough_mask, similarity_map, point_coords, point_labels = self.cmrs.generate_rough_mask(
#                 query_img, sup_imgs, sup_masks, self.sam2_predictor
#             )
#             results['rough_mask'] = rough_mask
#             results['similarity_map'] = similarity_map
        
#         if use_memory_refinement:
#             if not self._support_set:
#                 self.set_support(sup_imgs, sup_masks)
            
#             pred_mask, score = self.memory_module.predict_query(
#                 query_img, 
#                 rough_mask=rough_mask if use_cmrs else None,
#                 point_coords=point_coords,
#                 point_labels=point_labels
#             )
#             results['final_mask'] = pred_mask
#             results['score'] = score
#         else:
#             if rough_mask is not None:
#                 results['final_mask'] = rough_mask.astype(np.uint8)
#                 results['score'] = 1.0
#             else:
#                 H, W = query_img.size[::-1]
#                 results['final_mask'] = np.zeros((H, W), dtype=np.uint8)
#                 results['score'] = 0.0
        
#         return results


# def visualize_sp_sam_results(query_img, gt_mask, pred_results,
#                             support_images=None, support_masks=None,
#                             save_path=None, title="SP-SAM Results"):
#     import matplotlib.pyplot as plt
#     n_cols = 4
#     fig, axes = plt.subplots(1, n_cols, figsize=(16, 4))
    
#     axes[0].imshow(query_img)
#     axes[0].set_title("Query")
#     axes[0].axis('off')
    
#     axes[1].imshow(gt_mask, cmap='gray')
#     axes[1].set_title("GT")
#     axes[1].axis('off')
    
#     if pred_results.get('rough_mask') is not None:
#         axes[2].imshow(pred_results['rough_mask'], cmap='gray')
#         axes[2].set_title("Rough")
#     axes[2].axis('off')
    
#     if pred_results.get('final_mask') is not None:
#         axes[3].imshow(pred_results['final_mask'], cmap='gray')
#         if gt_mask is not None:
#             intersection = (pred_results['final_mask'] > 0) & (gt_mask > 0)
#             union = (pred_results['final_mask'] > 0) | (gt_mask > 0)
#             iou = intersection.sum() / (union.sum() + 1e-8)
#             axes[3].set_title(f"Final (IoU: {iou:.3f})")
#     axes[3].axis('off')
    
#     plt.suptitle(title)
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         plt.close()
#     else:
#         plt.show()


# if __name__ == '__main__':
#     print("=" * 60)
#     print("SP-SAM CMRS改进版")
#     print("=" * 60)
#     print(f"\n当前模式: {CMRS_MODE}")
#     print("\n可选模式:")
#     print("  'baseline'        - 原始CMRS")
#     print("  'dual_prototype'  - 前景+背景双原型")
#     print("  'multi_scale'     - 多尺度DINO特征")
#     print("  'box_prompt'      - 添加Box Prompt")
#     print("  'iterative_cmrs'  - 迭代CMRS")
#     print("  'dense_points'    - 更多点提示")





# """
# SP-SAM 最佳组合版
# ==================

# 基于实验结果，multi_scale是唯一有效的CMRS改进 (+0.26%)
# 尝试在multi_scale基础上叠加其他改进

# 组合方案：
# 1. multi_scale only (64.97%) - 当前最佳
# 2. multi_scale + dual_prototype
# 3. multi_scale + dense_points
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import cv2
# from PIL import Image
# from typing import List, Dict, Tuple, Optional, Any


# # ============================================================
# # 🔧 选择组合方案
# # ============================================================
# USE_MULTI_SCALE = True       # 多尺度特征 (必选)
# USE_DUAL_PROTOTYPE = False   # 前景+背景双原型
# USE_DENSE_POINTS = False     # 更多点提示 (20正+10负)

# # 推荐测试:
# # 1. 只开 USE_MULTI_SCALE = True (64.97%)
# # 2. 开 USE_MULTI_SCALE + USE_DUAL_PROTOTYPE 63.83%miou
# # 3. 开 USE_MULTI_SCALE + USE_DENSE_POINTS 62.25%miou
# # 4. 全开，60.55%miou

# class CMRSModule:
#     """最佳组合CMRS模块"""
    
#     def __init__(self, dino_model, dino_transform, device='cuda'):
#         self.dino_model = dino_model
#         self.dino_transform = dino_transform
#         self.device = device
#         self.current_prototype = None
#         self.current_bg_prototype = None
#         self.current_query_features = None
        
#         config_str = []
#         if USE_MULTI_SCALE:
#             config_str.append("multi_scale")
#         if USE_DUAL_PROTOTYPE:
#             config_str.append("dual_prototype")
#         if USE_DENSE_POINTS:
#             config_str.append("dense_points")
#         print(f"📌 CMRS配置: {' + '.join(config_str) if config_str else 'baseline'}")
        
#     def extract_dino_features(self, img_pil):
#         """标准单尺度特征提取"""
#         img_tensor = self.dino_transform(img_pil)[None].to(self.device)
#         with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
#             features = self.dino_model.get_intermediate_layers(img_tensor.to(torch.bfloat16))[0]
#             h = w = int(features.shape[1]**0.5)
#             feature_map = features.reshape(1, h, w, -1).permute(0, 3, 1, 2).float()
#         return feature_map.squeeze(0)
    
#     def extract_dino_features_multiscale(self, img_pil):
#         """多尺度特征提取"""
#         scales = [0.75, 1.0, 1.25]
#         original_size = img_pil.size
        
#         all_features = []
#         for scale in scales:
#             new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
#             img_scaled = img_pil.resize(new_size, Image.BILINEAR)
            
#             img_tensor = self.dino_transform(img_scaled)[None].to(self.device)
#             with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
#                 features = self.dino_model.get_intermediate_layers(img_tensor.to(torch.bfloat16))[0]
#                 h = w = int(features.shape[1]**0.5)
#                 feature_map = features.reshape(1, h, w, -1).permute(0, 3, 1, 2).float()
            
#             all_features.append(feature_map.squeeze(0))
        
#         # 统一到中间尺度的大小
#         target_h, target_w = all_features[1].shape[1], all_features[1].shape[2]
#         fused_features = []
#         for feat in all_features:
#             if feat.shape[1] != target_h or feat.shape[2] != target_w:
#                 feat = F.interpolate(feat.unsqueeze(0), size=(target_h, target_w), 
#                                     mode='bilinear', align_corners=False).squeeze(0)
#             fused_features.append(feat)
        
#         # 平均融合
#         fused = torch.stack(fused_features, dim=0).mean(dim=0)
#         return fused
    
#     def get_features(self, img_pil):
#         """根据配置选择特征提取方式"""
#         if USE_MULTI_SCALE:
#             return self.extract_dino_features_multiscale(img_pil)
#         else:
#             return self.extract_dino_features(img_pil)
    
#     def downsample_mask(self, mask_np, target_size):
#         h, w = target_size
#         resized_mask = cv2.resize(mask_np.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
#         resized_mask = (resized_mask > 0.5).astype(np.float32)
#         return torch.from_numpy(resized_mask).to(self.device)
    
#     def compute_prototype(self, features, mask):
#         """计算前景原型"""
#         mask_sum = mask.sum() + 1e-8
#         prototype = (features * mask.unsqueeze(0)).sum(dim=(1, 2)) / mask_sum
#         return prototype
    
#     def compute_bg_prototype(self, features, mask):
#         """计算背景原型"""
#         bg_mask = 1.0 - mask
#         bg_sum = bg_mask.sum() + 1e-8
#         bg_prototype = (features * bg_mask.unsqueeze(0)).sum(dim=(1, 2)) / bg_sum
#         return bg_prototype
    
#     def compute_similarity_map_dual_prototype(self, query_features, fg_prototype, bg_prototype):
#         """双原型对比"""
#         C, H, W = query_features.shape
        
#         fg_proto = F.normalize(fg_prototype, p=2, dim=0)
#         bg_proto = F.normalize(bg_prototype, p=2, dim=0)
        
#         query_flat = query_features.reshape(C, -1).T
#         query_flat = F.normalize(query_flat, p=2, dim=1)
        
#         fg_sim = torch.mv(query_flat, fg_proto)
#         bg_sim = torch.mv(query_flat, bg_proto)
        
#         # 对比分数
#         contrast_score = fg_sim - bg_sim
#         return contrast_score.reshape(H, W)
    
#     def compute_similarity_map_prototype(self, query_features, support_features_list, support_masks_list):
#         C, H, W = query_features.shape
        
#         fg_prototypes = []
#         bg_prototypes = []
        
#         for sup_feat, sup_mask in zip(support_features_list, support_masks_list):
#             fg_proto = self.compute_prototype(sup_feat, sup_mask)
#             fg_prototypes.append(fg_proto)
            
#             if USE_DUAL_PROTOTYPE:
#                 bg_proto = self.compute_bg_prototype(sup_feat, sup_mask)
#                 bg_prototypes.append(bg_proto)
        
#         avg_fg_prototype = torch.stack(fg_prototypes, dim=0).mean(dim=0)
#         self.current_prototype = F.normalize(avg_fg_prototype, p=2, dim=0)
        
#         if USE_DUAL_PROTOTYPE and bg_prototypes:
#             avg_bg_prototype = torch.stack(bg_prototypes, dim=0).mean(dim=0)
#             self.current_bg_prototype = F.normalize(avg_bg_prototype, p=2, dim=0)
            
#             similarity_map = self.compute_similarity_map_dual_prototype(
#                 query_features, self.current_prototype, self.current_bg_prototype
#             )
#         else:
#             query_flat = query_features.reshape(C, -1).T
#             query_flat = F.normalize(query_flat, p=2, dim=1)
#             similarity_scores = torch.mv(query_flat, self.current_prototype)
#             similarity_map = similarity_scores.reshape(H, W)
        
#         self.current_query_features = query_features
#         return similarity_map
    
#     def get_prompts_from_similarity(self, similarity_map, top_k=10, neg_k=5):
#         # Dense points模式
#         if USE_DENSE_POINTS:
#             top_k = 20
#             neg_k = 10
        
#         sim_np = similarity_map.cpu().numpy()
#         H, W = sim_np.shape
#         mean_sim, std_sim, max_sim = sim_np.mean(), sim_np.std(), sim_np.max()
        
#         if USE_DUAL_PROTOTYPE:
#             fg_threshold = max(0.1, mean_sim + 0.5 * std_sim)
#             bg_threshold = min(-0.1, mean_sim - 0.5 * std_sim)
#         else:
#             fg_threshold = max(mean_sim + 1.0 * std_sim, max_sim * 0.5)
#             bg_threshold = mean_sim - 0.5 * std_sim
        
#         fg_coords = np.argwhere(sim_np > fg_threshold)
        
#         if len(fg_coords) < top_k:
#             fg_threshold = mean_sim + 0.3 * std_sim if not USE_DUAL_PROTOTYPE else 0.0
#             fg_coords = np.argwhere(sim_np > fg_threshold)
        
#         if len(fg_coords) == 0:
#             flat_indices = np.argsort(sim_np.flatten())[-top_k:]
#             pos_points = np.array([np.unravel_index(idx, sim_np.shape) for idx in flat_indices])
#         else:
#             pos_points = self._sample_points_weighted_spatial(fg_coords, sim_np, top_k)
        
#         bg_coords = np.argwhere(sim_np < bg_threshold)
        
#         if len(bg_coords) >= neg_k:
#             neg_points = self._sample_negative_points_diverse(bg_coords, sim_np, neg_k, (H, W))
#         else:
#             neg_points = self._sample_from_borders(H, W, neg_k)
        
#         all_points = np.vstack([pos_points, neg_points])
#         all_labels = np.concatenate([np.ones(len(pos_points), dtype=np.int32),
#                                      np.zeros(len(neg_points), dtype=np.int32)])
#         return all_points[:, ::-1], all_labels
    
#     def _sample_points_weighted_spatial(self, coords, sim_np, num_points):
#         if len(coords) == 0:
#             return np.array([[sim_np.shape[0]//2, sim_np.shape[1]//2]])
        
#         sim_values = np.array([sim_np[y, x] for y, x in coords])
#         selected, selected_indices = [], []
#         first_idx = np.argmax(sim_values)
#         selected.append(coords[first_idx])
#         selected_indices.append(first_idx)
        
#         for _ in range(num_points - 1):
#             if len(selected_indices) >= len(coords):
#                 break
#             best_score, best_idx = -np.inf, -1
#             for i, (coord, sim_val) in enumerate(zip(coords, sim_values)):
#                 if i in selected_indices:
#                     continue
#                 min_dist = min(np.sqrt((coord[0]-s[0])**2 + (coord[1]-s[1])**2) for s in selected)
#                 score = sim_val * 0.7 + (min_dist / max(sim_np.shape)) * 0.3
#                 if score > best_score:
#                     best_score, best_idx = score, i
#             if best_idx >= 0:
#                 selected.append(coords[best_idx])
#                 selected_indices.append(best_idx)
#         return np.array(selected)
    
#     def _sample_negative_points_diverse(self, bg_coords, sim_np, num_points, shape):
#         H, W = shape
#         grid_h, grid_w = 3, 3
#         cell_h, cell_w = H // grid_h, W // grid_w
#         neg_points = []
#         for gh in range(grid_h):
#             for gw in range(grid_w):
#                 if len(neg_points) >= num_points:
#                     break
#                 y_start, y_end = gh * cell_h, (gh + 1) * cell_h
#                 x_start, x_end = gw * cell_w, (gw + 1) * cell_w
#                 cell_mask = ((bg_coords[:, 0] >= y_start) & (bg_coords[:, 0] < y_end) &
#                             (bg_coords[:, 1] >= x_start) & (bg_coords[:, 1] < x_end))
#                 cell_coords = bg_coords[cell_mask]
#                 if len(cell_coords) > 0:
#                     cell_sims = np.array([sim_np[y, x] for y, x in cell_coords])
#                     neg_points.append(cell_coords[np.argmin(cell_sims)])
#         if len(neg_points) < num_points and len(bg_coords) > 0:
#             indices = np.random.choice(len(bg_coords), min(num_points - len(neg_points), len(bg_coords)), replace=False)
#             for idx in indices:
#                 neg_points.append(bg_coords[idx])
#         return np.array(neg_points[:num_points]) if neg_points else np.array([[0, 0]])
    
#     def _sample_from_borders(self, H, W, num_points):
#         border = 3
#         corners = [[border, border], [border, W-border-1], [H-border-1, border], [H-border-1, W-border-1]]
#         neg_points = corners[:min(num_points, 4)]
#         while len(neg_points) < num_points:
#             side = np.random.randint(4)
#             if side == 0:
#                 neg_points.append([border, np.random.randint(border, W-border)])
#             elif side == 1:
#                 neg_points.append([H-border-1, np.random.randint(border, W-border)])
#             elif side == 2:
#                 neg_points.append([np.random.randint(border, H-border), border])
#             else:
#                 neg_points.append([np.random.randint(border, H-border), W-border-1])
#         return np.array(neg_points)
    
#     def generate_rough_mask(self, query_img, support_images, support_masks, sam2_predictor, top_k=10, neg_k=5):
#         query_features = self.get_features(query_img)
#         C, H_feat, W_feat = query_features.shape
        
#         support_features_list, support_masks_list = [], []
#         for sup_img, sup_mask in zip(support_images, support_masks):
#             sup_features = self.get_features(sup_img)
#             downsampled_mask = self.downsample_mask(sup_mask, (sup_features.shape[1], sup_features.shape[2]))
#             support_features_list.append(sup_features)
#             support_masks_list.append(downsampled_mask)
        
#         similarity_map = self.compute_similarity_map_prototype(query_features, support_features_list, support_masks_list)
#         point_coords_feat, point_labels = self.get_prompts_from_similarity(similarity_map, top_k, neg_k)
        
#         H_img, W_img = query_img.size[::-1]
#         H_feat, W_feat = similarity_map.shape
#         scale_y, scale_x = H_img / H_feat, W_img / W_feat
        
#         point_coords_img = point_coords_feat.copy().astype(np.float32)
#         point_coords_img[:, 0] *= scale_x
#         point_coords_img[:, 1] *= scale_y
#         point_coords_img = point_coords_img.astype(np.int32)
        
#         sam2_predictor.set_image(np.array(query_img))
        
#         try:
#             masks, scores, _ = sam2_predictor.predict(
#                 point_coords=point_coords_img, 
#                 point_labels=point_labels, 
#                 multimask_output=True
#             )
#             rough_mask = masks[np.argmax(scores)]
#         except Exception as e:
#             print(f"   ⚠️  SAM2预测失败: {e}")
#             rough_mask = None
        
#         return rough_mask, similarity_map.cpu().numpy(), point_coords_img, point_labels


# class SAM2MemoryModule:
#     """Memory模块 - A+B版本"""
    
#     def __init__(self, sam2_model, device='cuda', sam2_model_type='large'):
#         self.model = sam2_model
#         self.device = device
#         self.sam2_model_type = sam2_model_type
#         self.prev_out = {}
#         self.support_set = False
#         self.image_size = getattr(sam2_model, 'image_size', 1024)
#         self.num_feature_levels = getattr(sam2_model, 'num_feature_levels', 3)
    
#     def clear_support(self):
#         self.prev_out = {}
#         self.support_set = False
    
#     def _prepare_image(self, img):
#         target_size = self.image_size
#         img_resized = img.resize((target_size, target_size), Image.BILINEAR)
#         img_np = np.array(img_resized).astype(np.float32) / 255.0
#         img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
#         return img_tensor.unsqueeze(0).to(self.device)
    
#     def _prepare_mask(self, mask, target_size=None):
#         if target_size is None:
#             target_size = self.image_size
#         mask_resized = cv2.resize(mask.astype(np.float32), (target_size, target_size), interpolation=cv2.INTER_NEAREST)
#         mask_tensor = torch.from_numpy(mask_resized).float()
#         return mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
    
#     def encode_support(self, support_img, support_mask):
#         img_tensor = self._prepare_image(support_img)
        
#         with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
#             backbone_out = self.model.forward_image(img_tensor)
#             feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
            
#             for i in range(len(feature_maps)):
#                 feature_maps[i] = feature_maps[i].clone()
            
#             pix_feat = feature_maps[-1]
#             mask_tensor = self._prepare_mask(support_mask)
#             high_res_masks = mask_tensor.float() * 20.0 - 10.0
            
#             maskmem_out = self.model.memory_encoder(pix_feat=pix_feat, masks=high_res_masks, skip_mask_sigmoid=True)
            
#             self.prev_out.setdefault("maskmem_features", []).append(maskmem_out["vision_features"].clone())
#             if "maskmem_pos_enc" not in self.prev_out:
#                 self.prev_out["maskmem_pos_enc"] = [m.clone() for m in maskmem_out["vision_pos_enc"]]
        
#         self.support_set = True
    
#     def set_support(self, support_images, support_masks, target_size=None):
#         self.clear_support()
#         for img, mask in zip(support_images, support_masks):
#             self.encode_support(img, mask)
#         return self.support_set
    
#     def predict_query(self, query_img, rough_mask=None, point_coords=None, point_labels=None):
#         if not self.support_set:
#             raise ValueError("请先调用set_support()")
        
#         original_size = query_img.size[::-1]
#         img_tensor = self._prepare_image(query_img)
#         B = img_tensor.shape[0]
        
#         with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
#             backbone_out = self.model.forward_image(img_tensor)
#             feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
#             vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels:]
            
#             for i in range(len(feature_maps)):
#                 feature_maps[i] = feature_maps[i].clone()
#                 vision_pos_embeds[i] = vision_pos_embeds[i].clone()
            
#             high_res_features = feature_maps[:-1]
#             pix_feat = feature_maps[-1]
            
#             # Memory Attention
#             to_cat_memory = [m.flatten(2).permute(2, 0, 1) for m in self.prev_out["maskmem_features"]]
#             to_cat_memory_pos_embed = len(self.prev_out["maskmem_features"]) * [
#                 self.prev_out["maskmem_pos_enc"][0].flatten(2).permute(2, 0, 1)
#             ]
            
#             memory = torch.cat(to_cat_memory, dim=0)
#             memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)
            
#             pix_feat_with_mem = self.model.memory_attention(
#                 curr=pix_feat.flatten(2).permute(2, 0, 1),
#                 curr_pos=vision_pos_embeds[-1].flatten(2).permute(2, 0, 1),
#                 memory=memory, memory_pos=memory_pos_embed, num_obj_ptr_tokens=0
#             )
#             pix_feat_with_mem = pix_feat_with_mem.clone().permute(1, 2, 0).view(*pix_feat.shape)
            
#             # 点提示
#             if point_coords is not None and point_labels is not None:
#                 scale_x = self.image_size / original_size[1]
#                 scale_y = self.image_size / original_size[0]
                
#                 scaled_coords = point_coords.copy().astype(np.float32)
#                 scaled_coords[:, 0] *= scale_x
#                 scaled_coords[:, 1] *= scale_y
                
#                 sam_point_coords = torch.from_numpy(scaled_coords).unsqueeze(0).float().to(self.device)
#                 sam_point_labels = torch.from_numpy(point_labels.astype(np.int32)).unsqueeze(0).to(self.device)
#             else:
#                 sam_point_coords = torch.zeros(B, 1, 2, device=self.device)
#                 sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=self.device)
            
#             # Mask Prompt
#             sam_mask_prompt = None
#             if rough_mask is not None:
#                 mask_tensor = self._prepare_mask(rough_mask, target_size=self.image_size)
#                 high_res_mask_logits = mask_tensor.float() * 20.0 - 10.0
#                 sam_mask_prompt = F.interpolate(high_res_mask_logits, size=(256, 256),
#                                                 mode='bilinear', align_corners=False)
            
#             sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
#                 points=(sam_point_coords, sam_point_labels), boxes=None, masks=sam_mask_prompt
#             )
#             sparse_embeddings = sparse_embeddings.clone()
#             dense_embeddings = dense_embeddings.clone()
#             image_pe = self.model.sam_prompt_encoder.get_dense_pe().clone()
            
#             low_res_masks, ious, _, _ = self.model.sam_mask_decoder(
#                 image_embeddings=pix_feat_with_mem, image_pe=image_pe,
#                 sparse_prompt_embeddings=sparse_embeddings,
#                 dense_prompt_embeddings=dense_embeddings,
#                 multimask_output=True,
#                 repeat_image=False, high_res_features=high_res_features
#             )
            
#             best_idx = torch.argmax(ious[0])
#             best_mask = low_res_masks[0, best_idx:best_idx+1].unsqueeze(0)
#             best_iou = ious[0, best_idx]
            
#             best_mask = best_mask.float()
#             high_res_masks = F.interpolate(best_mask, size=(self.image_size, self.image_size),
#                                            mode='bilinear', align_corners=False)
#             final_masks = F.interpolate(high_res_masks, size=original_size,
#                                         mode='bilinear', align_corners=False)
            
#             pred_mask = (final_masks[0, 0] > 0).cpu().numpy().astype(np.uint8)
#             score = float(best_iou.cpu())
        
#         return pred_mask, score


# class SPSAMModel:
#     def __init__(self, sam2_model, sam2_predictor, dino_model, dino_transform,
#                  device='cuda', sam2_model_type='large'):
#         self.sam2_model = sam2_model
#         self.sam2_predictor = sam2_predictor
#         self.device = device
        
#         self.cmrs = CMRSModule(dino_model, dino_transform, device)
#         self.memory_module = SAM2MemoryModule(sam2_model, device, sam2_model_type)
        
#         self._support_images = []
#         self._support_masks = []
#         self._support_set = False
        
#         print(f"✅ SPSAMModel 最佳组合版")
    
#     def set_support(self, support_images, support_masks):
#         self._support_images = support_images
#         self._support_masks = support_masks
#         success = self.memory_module.set_support(support_images, support_masks)
#         self._support_set = success
#         return success
    
#     def clear_support(self):
#         self._support_images = []
#         self._support_masks = []
#         self._support_set = False
#         self.memory_module.clear_support()
    
#     def predict(self, query_img, support_images, support_masks, use_cmrs=True, use_memory_refinement=False):
#         if use_memory_refinement:
#             self.set_support(support_images, support_masks)
#         return self.predict_query(query_img, use_cmrs, use_memory_refinement, support_images, support_masks)
    
#     def predict_query(self, query_img, use_cmrs=True, use_memory_refinement=False,
#                      support_images=None, support_masks=None):
#         sup_imgs = support_images if support_images else self._support_images
#         sup_masks = support_masks if support_masks else self._support_masks
        
#         if len(sup_imgs) == 0:
#             raise ValueError("No support samples")
        
#         results = {'final_mask': None, 'rough_mask': None, 'similarity_map': None, 'score': 0.0}
        
#         rough_mask = None
#         point_coords = None
#         point_labels = None
        
#         if use_cmrs:
#             rough_mask, similarity_map, point_coords, point_labels = self.cmrs.generate_rough_mask(
#                 query_img, sup_imgs, sup_masks, self.sam2_predictor
#             )
#             results['rough_mask'] = rough_mask
#             results['similarity_map'] = similarity_map
        
#         if use_memory_refinement:
#             if not self._support_set:
#                 self.set_support(sup_imgs, sup_masks)
            
#             pred_mask, score = self.memory_module.predict_query(
#                 query_img, 
#                 rough_mask=rough_mask if use_cmrs else None,
#                 point_coords=point_coords,
#                 point_labels=point_labels
#             )
#             results['final_mask'] = pred_mask
#             results['score'] = score
#         else:
#             if rough_mask is not None:
#                 results['final_mask'] = rough_mask.astype(np.uint8)
#                 results['score'] = 1.0
#             else:
#                 H, W = query_img.size[::-1]
#                 results['final_mask'] = np.zeros((H, W), dtype=np.uint8)
#                 results['score'] = 0.0
        
#         return results


# def visualize_sp_sam_results(query_img, gt_mask, pred_results,
#                             support_images=None, support_masks=None,
#                             save_path=None, title="SP-SAM Results"):
#     import matplotlib.pyplot as plt
#     n_cols = 4
#     fig, axes = plt.subplots(1, n_cols, figsize=(16, 4))
    
#     axes[0].imshow(query_img)
#     axes[0].set_title("Query")
#     axes[0].axis('off')
    
#     axes[1].imshow(gt_mask, cmap='gray')
#     axes[1].set_title("GT")
#     axes[1].axis('off')
    
#     if pred_results.get('rough_mask') is not None:
#         axes[2].imshow(pred_results['rough_mask'], cmap='gray')
#         axes[2].set_title("Rough")
#     axes[2].axis('off')
    
#     if pred_results.get('final_mask') is not None:
#         axes[3].imshow(pred_results['final_mask'], cmap='gray')
#         if gt_mask is not None:
#             intersection = (pred_results['final_mask'] > 0) & (gt_mask > 0)
#             union = (pred_results['final_mask'] > 0) | (gt_mask > 0)
#             iou = intersection.sum() / (union.sum() + 1e-8)
#             axes[3].set_title(f"Final (IoU: {iou:.3f})")
#     axes[3].axis('off')
    
#     plt.suptitle(title)
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         plt.close()
#     else:
#         plt.show()


# if __name__ == '__main__':
#     print("=" * 60)
#     print("SP-SAM 最佳组合版")
#     print("=" * 60)
#     print(f"\n当前配置:")
#     print(f"  USE_MULTI_SCALE = {USE_MULTI_SCALE}")
#     print(f"  USE_DUAL_PROTOTYPE = {USE_DUAL_PROTOTYPE}")
#     print(f"  USE_DENSE_POINTS = {USE_DENSE_POINTS}")




# """
# SP-SAM 模型多层特征版本
# ========================

# 将图像多尺度（Image Pyramid）改为模型多层特征（Multi-layer Features）
# 从DINO的不同层提取特征并融合，而不是对图像进行多次缩放

# 配置选项：
# - USE_MULTI_LAYER: 使用模型多层特征
# - USE_MULTI_SCALE: 使用图像多尺度（原方案）
# - USE_DUAL_PROTOTYPE: 前景+背景双原型
# - USE_DENSE_POINTS: 更多点提示
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import cv2
# from PIL import Image
# from typing import List, Dict, Tuple, Optional, Any


# # ============================================================
# # 🔧 配置选项
# # ============================================================
# USE_MULTI_LAYER = False       # 模型多层特征（新方案）
# USE_MULTI_SCALE = True      # 图像多尺度（原方案，与MULTI_LAYER二选一）
# USE_DUAL_PROTOTYPE = False   # 前景+背景双原型
# USE_DENSE_POINTS = False     # 更多点提示 (20正+10负)

# # 多层特征配置
# DINO_LAYERS = [8, 16, 24]    # 要提取的层数（针对DINOv2-Large 24层）
# LAYER_FUSION = 'weighted'        # 融合方式: 'mean', 'concat', 'weighted'
# LAYER_WEIGHTS = [0.2, 0.3, 0.5]  # 加权融合时的权重（深层权重更高）


# class CMRSModule:
#     """
#     CMRS模块 - 支持模型多层特征提取
    
#     新增功能：
#     - extract_dino_features_multilayer: 从DINO不同层提取特征
#     - 支持多种融合策略: mean, concat, weighted
#     """
    
#     def __init__(self, dino_model, dino_transform, device='cuda'):
#         self.dino_model = dino_model
#         self.dino_transform = dino_transform
#         self.device = device
#         self.current_prototype = None
#         self.current_bg_prototype = None
#         self.current_query_features = None
        
#         # 如果使用concat融合，需要记录通道数变化
#         self.feature_channels = None
        
#         # 打印配置
#         config_str = []
#         if USE_MULTI_LAYER:
#             config_str.append(f"multi_layer(layers={DINO_LAYERS}, fusion={LAYER_FUSION})")
#         if USE_MULTI_SCALE:
#             config_str.append("multi_scale")
#         if USE_DUAL_PROTOTYPE:
#             config_str.append("dual_prototype")
#         if USE_DENSE_POINTS:
#             config_str.append("dense_points")
#         print(f"📌 CMRS配置: {' + '.join(config_str) if config_str else 'baseline'}")
    
#     # ============================================================
#     # 特征提取方法
#     # ============================================================
    
#     def extract_dino_features(self, img_pil):
#         """标准单层特征提取（baseline）"""
#         img_tensor = self.dino_transform(img_pil)[None].to(self.device)
        
#         with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
#             features = self.dino_model.get_intermediate_layers(
#                 img_tensor.to(torch.bfloat16)
#             )[0]
#             h = w = int(features.shape[1] ** 0.5)
#             feature_map = features.reshape(1, h, w, -1).permute(0, 3, 1, 2).float()
        
#         return feature_map.squeeze(0)  # (C, H, W)
    
#     def extract_dino_features_multiscale(self, img_pil):
#         """图像多尺度特征提取（原方案）"""
#         scales = [0.75, 1.0, 1.25]
#         original_size = img_pil.size
        
#         all_features = []
#         for scale in scales:
#             new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
#             img_scaled = img_pil.resize(new_size, Image.BILINEAR)
            
#             img_tensor = self.dino_transform(img_scaled)[None].to(self.device)
#             with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
#                 features = self.dino_model.get_intermediate_layers(
#                     img_tensor.to(torch.bfloat16)
#                 )[0]
#                 h = w = int(features.shape[1] ** 0.5)
#                 feature_map = features.reshape(1, h, w, -1).permute(0, 3, 1, 2).float()
            
#             all_features.append(feature_map.squeeze(0))
        
#         # 统一到中间尺度的大小
#         target_h, target_w = all_features[1].shape[1], all_features[1].shape[2]
#         fused_features = []
#         for feat in all_features:
#             if feat.shape[1] != target_h or feat.shape[2] != target_w:
#                 feat = F.interpolate(feat.unsqueeze(0), size=(target_h, target_w),
#                                     mode='bilinear', align_corners=False).squeeze(0)
#             fused_features.append(feat)
        
#         # 平均融合
#         fused = torch.stack(fused_features, dim=0).mean(dim=0)
#         return fused
    
#     def extract_dino_features_multilayer(self, img_pil):
#         """模型多层特征提取（自动适配层数）"""
#         img_tensor = self.dino_transform(img_pil)[None].to(self.device)
        
#         with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
#             # 自动获取模型总层数
#             total_layers = len(self.dino_model.blocks)
            
#             # 根据总层数自动选择要提取的层
#             if total_layers >= 24:
#                 layers_to_extract = [7, 15, 23]
#             elif total_layers >= 12:
#                 layers_to_extract = [1, 6, 11]
#             elif total_layers >= 6:
#                 layers_to_extract = [1, 3, 5]
#             else:
#                 # 层数太少，只用最后一层
#                 layers_to_extract = [total_layers]
            
#             # 确保不超过总层数
#             layers_to_extract = [min(l, total_layers) for l in layers_to_extract]
#             layers_to_extract = list(set(layers_to_extract))  # 去重
#             layers_to_extract.sort()
            
#             print(f"📌 DINO总层数: {total_layers}, 提取层: {layers_to_extract}")
            
#             # 提取多个中间层的特征
#             features_list = self.dino_model.get_intermediate_layers(
#                 img_tensor.to(torch.bfloat16),
#                 n=layers_to_extract,
#                 reshape=False
#             )
            
#             all_feature_maps = []
#             for features in features_list:
#                 h = w = int(features.shape[1] ** 0.5)
#                 feature_map = features.reshape(1, h, w, -1).permute(0, 3, 1, 2).float()
#                 all_feature_maps.append(feature_map.squeeze(0))
            
#             # 融合
#             if len(all_feature_maps) == 1:
#                 fused = all_feature_maps[0]
#             elif LAYER_FUSION == 'mean':
#                 fused = torch.stack(all_feature_maps, dim=0).mean(dim=0)
#             elif LAYER_FUSION == 'concat':
#                 fused = torch.cat(all_feature_maps, dim=0)
#             elif LAYER_FUSION == 'weighted':
#                 weights = LAYER_WEIGHTS[:len(all_feature_maps)]
#                 weights = [w / sum(weights) for w in weights]  # 归一化
#                 fused = sum(w * f for w, f in zip(weights, all_feature_maps))
#             elif LAYER_FUSION == 'max':
#                 fused = torch.stack(all_feature_maps, dim=0).max(dim=0)[0]
#             else:
#                 fused = torch.stack(all_feature_maps, dim=0).mean(dim=0)
        
#         self.feature_channels = fused.shape[0]
#         return fused
    
#     def get_features(self, img_pil):
#         """根据配置选择特征提取方式"""
#         if USE_MULTI_LAYER:
#             return self.extract_dino_features_multilayer(img_pil)
#         elif USE_MULTI_SCALE:
#             return self.extract_dino_features_multiscale(img_pil)
#         else:
#             return self.extract_dino_features(img_pil)
    
#     # ============================================================
#     # Mask处理
#     # ============================================================
    
#     def downsample_mask(self, mask_np, target_size):
#         """下采样mask到特征图尺寸"""
#         h, w = target_size
#         resized_mask = cv2.resize(mask_np.astype(np.float32), (w, h),
#                                   interpolation=cv2.INTER_NEAREST)
#         resized_mask = (resized_mask > 0.5).astype(np.float32)
#         return torch.from_numpy(resized_mask).to(self.device)
    
#     # ============================================================
#     # Prototype计算
#     # ============================================================
    
#     def compute_prototype(self, features, mask):
#         """计算前景原型（Masked Average Pooling）"""
#         mask_sum = mask.sum() + 1e-8
#         prototype = (features * mask.unsqueeze(0)).sum(dim=(1, 2)) / mask_sum
#         return prototype
    
#     def compute_bg_prototype(self, features, mask):
#         """计算背景原型"""
#         bg_mask = 1.0 - mask
#         bg_sum = bg_mask.sum() + 1e-8
#         bg_prototype = (features * bg_mask.unsqueeze(0)).sum(dim=(1, 2)) / bg_sum
#         return bg_prototype
    
#     def compute_similarity_map_dual_prototype(self, query_features, fg_prototype, bg_prototype):
#         """双原型对比计算相似度"""
#         C, H, W = query_features.shape
        
#         fg_proto = F.normalize(fg_prototype, p=2, dim=0)
#         bg_proto = F.normalize(bg_prototype, p=2, dim=0)
        
#         query_flat = query_features.reshape(C, -1).T
#         query_flat = F.normalize(query_flat, p=2, dim=1)
        
#         fg_sim = torch.mv(query_flat, fg_proto)
#         bg_sim = torch.mv(query_flat, bg_proto)
        
#         # 对比分数 = 前景相似度 - 背景相似度
#         contrast_score = fg_sim - bg_sim
#         return contrast_score.reshape(H, W)
    
#     def compute_similarity_map_prototype(self, query_features, support_features_list, support_masks_list):
#         """计算query与support原型的相似度图"""
#         C, H, W = query_features.shape
        
#         fg_prototypes = []
#         bg_prototypes = []
        
#         for sup_feat, sup_mask in zip(support_features_list, support_masks_list):
#             fg_proto = self.compute_prototype(sup_feat, sup_mask)
#             fg_prototypes.append(fg_proto)
            
#             if USE_DUAL_PROTOTYPE:
#                 bg_proto = self.compute_bg_prototype(sup_feat, sup_mask)
#                 bg_prototypes.append(bg_proto)
        
#         # 平均所有support的前景原型
#         avg_fg_prototype = torch.stack(fg_prototypes, dim=0).mean(dim=0)
#         self.current_prototype = F.normalize(avg_fg_prototype, p=2, dim=0)
        
#         if USE_DUAL_PROTOTYPE and bg_prototypes:
#             avg_bg_prototype = torch.stack(bg_prototypes, dim=0).mean(dim=0)
#             self.current_bg_prototype = F.normalize(avg_bg_prototype, p=2, dim=0)
            
#             similarity_map = self.compute_similarity_map_dual_prototype(
#                 query_features, self.current_prototype, self.current_bg_prototype
#             )
#         else:
#             # 标准余弦相似度
#             query_flat = query_features.reshape(C, -1).T
#             query_flat = F.normalize(query_flat, p=2, dim=1)
#             similarity_scores = torch.mv(query_flat, self.current_prototype)
#             similarity_map = similarity_scores.reshape(H, W)
        
#         self.current_query_features = query_features
#         return similarity_map
    
#     # ============================================================
#     # 点提示采样
#     # ============================================================
    
#     def get_prompts_from_similarity(self, similarity_map, top_k=10, neg_k=5):
#         """从相似度图采样正负点提示"""
#         # Dense points模式
#         if USE_DENSE_POINTS:
#             top_k = 20
#             neg_k = 10
        
#         sim_np = similarity_map.cpu().numpy()
#         H, W = sim_np.shape
#         mean_sim, std_sim, max_sim = sim_np.mean(), sim_np.std(), sim_np.max()
        
#         # 阈值设置
#         if USE_DUAL_PROTOTYPE:
#             fg_threshold = max(0.1, mean_sim + 0.5 * std_sim)
#             bg_threshold = min(-0.1, mean_sim - 0.5 * std_sim)
#         else:
#             fg_threshold = max(mean_sim + 1.0 * std_sim, max_sim * 0.5)
#             bg_threshold = mean_sim - 0.5 * std_sim
        
#         # 正样本采样
#         fg_coords = np.argwhere(sim_np > fg_threshold)
        
#         if len(fg_coords) < top_k:
#             fg_threshold = mean_sim + 0.3 * std_sim if not USE_DUAL_PROTOTYPE else 0.0
#             fg_coords = np.argwhere(sim_np > fg_threshold)
        
#         if len(fg_coords) == 0:
#             flat_indices = np.argsort(sim_np.flatten())[-top_k:]
#             pos_points = np.array([np.unravel_index(idx, sim_np.shape) for idx in flat_indices])
#         else:
#             pos_points = self._sample_points_weighted_spatial(fg_coords, sim_np, top_k)
        
#         # 负样本采样
#         bg_coords = np.argwhere(sim_np < bg_threshold)
        
#         if len(bg_coords) >= neg_k:
#             neg_points = self._sample_negative_points_diverse(bg_coords, sim_np, neg_k, (H, W))
#         else:
#             neg_points = self._sample_from_borders(H, W, neg_k)
        
#         # 合并
#         all_points = np.vstack([pos_points, neg_points])
#         all_labels = np.concatenate([np.ones(len(pos_points), dtype=np.int32),
#                                      np.zeros(len(neg_points), dtype=np.int32)])
#         return all_points[:, ::-1], all_labels  # (y,x) → (x,y)
    
#     def _sample_points_weighted_spatial(self, coords, sim_np, num_points):
#         """加权空间分散采样正样本"""
#         if len(coords) == 0:
#             return np.array([[sim_np.shape[0]//2, sim_np.shape[1]//2]])
        
#         sim_values = np.array([sim_np[y, x] for y, x in coords])
#         selected, selected_indices = [], []
        
#         # 第一个点：相似度最高
#         first_idx = np.argmax(sim_values)
#         selected.append(coords[first_idx])
#         selected_indices.append(first_idx)
        
#         # 后续点：平衡相似度和空间分散
#         for _ in range(num_points - 1):
#             if len(selected_indices) >= len(coords):
#                 break
#             best_score, best_idx = -np.inf, -1
#             for i, (coord, sim_val) in enumerate(zip(coords, sim_values)):
#                 if i in selected_indices:
#                     continue
#                 min_dist = min(np.sqrt((coord[0]-s[0])**2 + (coord[1]-s[1])**2) for s in selected)
#                 # 70%相似度 + 30%空间分散
#                 score = sim_val * 0.7 + (min_dist / max(sim_np.shape)) * 0.3
#                 if score > best_score:
#                     best_score, best_idx = score, i
#             if best_idx >= 0:
#                 selected.append(coords[best_idx])
#                 selected_indices.append(best_idx)
        
#         return np.array(selected)
    
#     def _sample_negative_points_diverse(self, bg_coords, sim_np, num_points, shape):
#         """3×3网格分散采样负样本"""
#         H, W = shape
#         grid_h, grid_w = 3, 3
#         cell_h, cell_w = H // grid_h, W // grid_w
#         neg_points = []
        
#         for gh in range(grid_h):
#             for gw in range(grid_w):
#                 if len(neg_points) >= num_points:
#                     break
#                 y_start, y_end = gh * cell_h, (gh + 1) * cell_h
#                 x_start, x_end = gw * cell_w, (gw + 1) * cell_w
#                 cell_mask = ((bg_coords[:, 0] >= y_start) & (bg_coords[:, 0] < y_end) &
#                             (bg_coords[:, 1] >= x_start) & (bg_coords[:, 1] < x_end))
#                 cell_coords = bg_coords[cell_mask]
#                 if len(cell_coords) > 0:
#                     cell_sims = np.array([sim_np[y, x] for y, x in cell_coords])
#                     neg_points.append(cell_coords[np.argmin(cell_sims)])
        
#         if len(neg_points) < num_points and len(bg_coords) > 0:
#             indices = np.random.choice(len(bg_coords), 
#                                       min(num_points - len(neg_points), len(bg_coords)), 
#                                       replace=False)
#             for idx in indices:
#                 neg_points.append(bg_coords[idx])
        
#         return np.array(neg_points[:num_points]) if neg_points else np.array([[0, 0]])
    
#     def _sample_from_borders(self, H, W, num_points):
#         """从边界采样负样本（保底策略）"""
#         border = 3
#         corners = [[border, border], [border, W-border-1], 
#                    [H-border-1, border], [H-border-1, W-border-1]]
#         neg_points = corners[:min(num_points, 4)]
        
#         while len(neg_points) < num_points:
#             side = np.random.randint(4)
#             if side == 0:
#                 neg_points.append([border, np.random.randint(border, W-border)])
#             elif side == 1:
#                 neg_points.append([H-border-1, np.random.randint(border, W-border)])
#             elif side == 2:
#                 neg_points.append([np.random.randint(border, H-border), border])
#             else:
#                 neg_points.append([np.random.randint(border, H-border), W-border-1])
        
#         return np.array(neg_points)
    
#     # ============================================================
#     # 主函数：生成粗分割
#     # ============================================================
    
#     def generate_rough_mask(self, query_img, support_images, support_masks, 
#                            sam2_predictor, top_k=10, neg_k=5):
#         """
#         生成粗分割掩码
        
#         流程：
#         1. 提取query和support的特征
#         2. 计算相似度图
#         3. 采样点提示
#         4. SAM2生成粗分割
#         """
#         # 1. 提取query特征
#         query_features = self.get_features(query_img)
#         C, H_feat, W_feat = query_features.shape
        
#         # 2. 提取所有support的特征和下采样mask
#         support_features_list, support_masks_list = [], []
#         for sup_img, sup_mask in zip(support_images, support_masks):
#             sup_features = self.get_features(sup_img)
#             downsampled_mask = self.downsample_mask(sup_mask, 
#                                                     (sup_features.shape[1], sup_features.shape[2]))
#             support_features_list.append(sup_features)
#             support_masks_list.append(downsampled_mask)
        
#         # 3. 计算相似度图
#         similarity_map = self.compute_similarity_map_prototype(
#             query_features, support_features_list, support_masks_list
#         )
        
#         # 4. 采样点提示
#         point_coords_feat, point_labels = self.get_prompts_from_similarity(
#             similarity_map, top_k, neg_k
#         )
        
#         # 5. 坐标缩放：特征图尺度 → 原图尺度
#         H_img, W_img = query_img.size[::-1]
#         H_feat, W_feat = similarity_map.shape
#         scale_y, scale_x = H_img / H_feat, W_img / W_feat
        
#         point_coords_img = point_coords_feat.copy().astype(np.float32)
#         point_coords_img[:, 0] *= scale_x
#         point_coords_img[:, 1] *= scale_y
#         point_coords_img = point_coords_img.astype(np.int32)
        
#         # 6. SAM2预测
#         sam2_predictor.set_image(np.array(query_img))
        
#         try:
#             masks, scores, _ = sam2_predictor.predict(
#                 point_coords=point_coords_img,
#                 point_labels=point_labels,
#                 multimask_output=True
#             )
#             rough_mask = masks[np.argmax(scores)]
#         except Exception as e:
#             print(f"   ⚠️  SAM2预测失败: {e}")
#             rough_mask = None
        
#         return rough_mask, similarity_map.cpu().numpy(), point_coords_img, point_labels


# class SAM2MemoryModule:
#     """Memory模块 - 用于精炼分割结果"""
    
#     def __init__(self, sam2_model, device='cuda', sam2_model_type='large'):
#         self.model = sam2_model
#         self.device = device
#         self.sam2_model_type = sam2_model_type
#         self.prev_out = {}
#         self.support_set = False
#         self.image_size = getattr(sam2_model, 'image_size', 1024)
#         self.num_feature_levels = getattr(sam2_model, 'num_feature_levels', 3)
    
#     def clear_support(self):
#         self.prev_out = {}
#         self.support_set = False
    
#     def _prepare_image(self, img):
#         target_size = self.image_size
#         img_resized = img.resize((target_size, target_size), Image.BILINEAR)
#         img_np = np.array(img_resized).astype(np.float32) / 255.0
#         img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
#         return img_tensor.unsqueeze(0).to(self.device)
    
#     def _prepare_mask(self, mask, target_size=None):
#         if target_size is None:
#             target_size = self.image_size
#         mask_resized = cv2.resize(mask.astype(np.float32), (target_size, target_size),
#                                   interpolation=cv2.INTER_NEAREST)
#         mask_tensor = torch.from_numpy(mask_resized).float()
#         return mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
    
#     def encode_support(self, support_img, support_mask):
#         """编码support样本到memory"""
#         img_tensor = self._prepare_image(support_img)
        
#         with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
#             backbone_out = self.model.forward_image(img_tensor)
#             feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
            
#             for i in range(len(feature_maps)):
#                 feature_maps[i] = feature_maps[i].clone()
            
#             pix_feat = feature_maps[-1]
#             mask_tensor = self._prepare_mask(support_mask)
#             high_res_masks = mask_tensor.float() * 20.0 - 10.0
            
#             maskmem_out = self.model.memory_encoder(
#                 pix_feat=pix_feat, masks=high_res_masks, skip_mask_sigmoid=True
#             )
            
#             self.prev_out.setdefault("maskmem_features", []).append(
#                 maskmem_out["vision_features"].clone()
#             )
#             if "maskmem_pos_enc" not in self.prev_out:
#                 self.prev_out["maskmem_pos_enc"] = [
#                     m.clone() for m in maskmem_out["vision_pos_enc"]
#                 ]
        
#         self.support_set = True
    
#     def set_support(self, support_images, support_masks, target_size=None):
#         """设置support集合"""
#         self.clear_support()
#         for img, mask in zip(support_images, support_masks):
#             self.encode_support(img, mask)
#         return self.support_set
    
#     def predict_query(self, query_img, rough_mask=None, point_coords=None, point_labels=None):
#         """预测query图像的分割"""
#         if not self.support_set:
#             raise ValueError("请先调用set_support()")
        
#         original_size = query_img.size[::-1]
#         img_tensor = self._prepare_image(query_img)
#         B = img_tensor.shape[0]
        
#         with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
#             backbone_out = self.model.forward_image(img_tensor)
#             feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
#             vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels:]
            
#             for i in range(len(feature_maps)):
#                 feature_maps[i] = feature_maps[i].clone()
#                 vision_pos_embeds[i] = vision_pos_embeds[i].clone()
            
#             high_res_features = feature_maps[:-1]
#             pix_feat = feature_maps[-1]
            
#             # Memory Attention
#             to_cat_memory = [m.flatten(2).permute(2, 0, 1) 
#                             for m in self.prev_out["maskmem_features"]]
#             to_cat_memory_pos_embed = len(self.prev_out["maskmem_features"]) * [
#                 self.prev_out["maskmem_pos_enc"][0].flatten(2).permute(2, 0, 1)
#             ]
            
#             memory = torch.cat(to_cat_memory, dim=0)
#             memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)
            
#             pix_feat_with_mem = self.model.memory_attention(
#                 curr=pix_feat.flatten(2).permute(2, 0, 1),
#                 curr_pos=vision_pos_embeds[-1].flatten(2).permute(2, 0, 1),
#                 memory=memory, memory_pos=memory_pos_embed, num_obj_ptr_tokens=0
#             )
#             pix_feat_with_mem = pix_feat_with_mem.clone().permute(1, 2, 0).view(*pix_feat.shape)
            
#             # 点提示
#             if point_coords is not None and point_labels is not None:
#                 scale_x = self.image_size / original_size[1]
#                 scale_y = self.image_size / original_size[0]
                
#                 scaled_coords = point_coords.copy().astype(np.float32)
#                 scaled_coords[:, 0] *= scale_x
#                 scaled_coords[:, 1] *= scale_y
                
#                 sam_point_coords = torch.from_numpy(scaled_coords).unsqueeze(0).float().to(self.device)
#                 sam_point_labels = torch.from_numpy(point_labels.astype(np.int32)).unsqueeze(0).to(self.device)
#             else:
#                 sam_point_coords = torch.zeros(B, 1, 2, device=self.device)
#                 sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=self.device)
            
#             # Mask Prompt
#             sam_mask_prompt = None
#             if rough_mask is not None:
#                 mask_tensor = self._prepare_mask(rough_mask, target_size=self.image_size)
#                 high_res_mask_logits = mask_tensor.float() * 20.0 - 10.0
#                 sam_mask_prompt = F.interpolate(high_res_mask_logits, size=(256, 256),
#                                                 mode='bilinear', align_corners=False)
            
#             sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
#                 points=(sam_point_coords, sam_point_labels), boxes=None, masks=sam_mask_prompt
#             )
#             sparse_embeddings = sparse_embeddings.clone()
#             dense_embeddings = dense_embeddings.clone()
#             image_pe = self.model.sam_prompt_encoder.get_dense_pe().clone()
            
#             low_res_masks, ious, _, _ = self.model.sam_mask_decoder(
#                 image_embeddings=pix_feat_with_mem, image_pe=image_pe,
#                 sparse_prompt_embeddings=sparse_embeddings,
#                 dense_prompt_embeddings=dense_embeddings,
#                 multimask_output=True,
#                 repeat_image=False, high_res_features=high_res_features
#             )
            
#             best_idx = torch.argmax(ious[0])
#             best_mask = low_res_masks[0, best_idx:best_idx+1].unsqueeze(0)
#             best_iou = ious[0, best_idx]
            
#             best_mask = best_mask.float()
#             high_res_masks = F.interpolate(best_mask, size=(self.image_size, self.image_size),
#                                            mode='bilinear', align_corners=False)
#             final_masks = F.interpolate(high_res_masks, size=original_size,
#                                         mode='bilinear', align_corners=False)
            
#             pred_mask = (final_masks[0, 0] > 0).cpu().numpy().astype(np.uint8)
#             score = float(best_iou.cpu())
        
#         return pred_mask, score


# class SPSAMModel:
#     """SP-SAM主模型"""
    
#     def __init__(self, sam2_model, sam2_predictor, dino_model, dino_transform,
#                  device='cuda', sam2_model_type='large'):
#         self.sam2_model = sam2_model
#         self.sam2_predictor = sam2_predictor
#         self.device = device
        
#         self.cmrs = CMRSModule(dino_model, dino_transform, device)
#         self.memory_module = SAM2MemoryModule(sam2_model, device, sam2_model_type)
        
#         self._support_images = []
#         self._support_masks = []
#         self._support_set = False
        
#         print(f"✅ SPSAMModel 初始化完成")
    
#     def set_support(self, support_images, support_masks):
#         """设置support集合"""
#         self._support_images = support_images
#         self._support_masks = support_masks
#         success = self.memory_module.set_support(support_images, support_masks)
#         self._support_set = success
#         return success
    
#     def clear_support(self):
#         """清除support集合"""
#         self._support_images = []
#         self._support_masks = []
#         self._support_set = False
#         self.memory_module.clear_support()
    
#     def predict(self, query_img, support_images, support_masks, 
#                 use_cmrs=True, use_memory_refinement=False):
#         """预测接口"""
#         if use_memory_refinement:
#             self.set_support(support_images, support_masks)
#         return self.predict_query(query_img, use_cmrs, use_memory_refinement, 
#                                   support_images, support_masks)
    
#     def predict_query(self, query_img, use_cmrs=True, use_memory_refinement=False,
#                      support_images=None, support_masks=None):
#         """预测query图像的分割"""
#         sup_imgs = support_images if support_images else self._support_images
#         sup_masks = support_masks if support_masks else self._support_masks
        
#         if len(sup_imgs) == 0:
#             raise ValueError("No support samples")
        
#         results = {'final_mask': None, 'rough_mask': None, 
#                   'similarity_map': None, 'score': 0.0}
        
#         rough_mask = None
#         point_coords = None
#         point_labels = None
        
#         # Stage 1: CMRS生成粗分割
#         if use_cmrs:
#             rough_mask, similarity_map, point_coords, point_labels = \
#                 self.cmrs.generate_rough_mask(
#                     query_img, sup_imgs, sup_masks, self.sam2_predictor
#                 )
#             results['rough_mask'] = rough_mask
#             results['similarity_map'] = similarity_map
        
#         # Stage 2: Memory模块精炼
#         if use_memory_refinement:
#             if not self._support_set:
#                 self.set_support(sup_imgs, sup_masks)
            
#             pred_mask, score = self.memory_module.predict_query(
#                 query_img,
#                 rough_mask=rough_mask if use_cmrs else None,
#                 point_coords=point_coords,
#                 point_labels=point_labels
#             )
#             results['final_mask'] = pred_mask
#             results['score'] = score
#         else:
#             if rough_mask is not None:
#                 results['final_mask'] = rough_mask.astype(np.uint8)
#                 results['score'] = 1.0
#             else:
#                 H, W = query_img.size[::-1]
#                 results['final_mask'] = np.zeros((H, W), dtype=np.uint8)
#                 results['score'] = 0.0
        
#         return results


# def visualize_sp_sam_results(query_img, gt_mask, pred_results,
#                             support_images=None, support_masks=None,
#                             save_path=None, title="SP-SAM Results"):
#     """可视化结果"""
#     import matplotlib.pyplot as plt
    
#     n_cols = 4
#     fig, axes = plt.subplots(1, n_cols, figsize=(16, 4))
    
#     axes[0].imshow(query_img)
#     axes[0].set_title("Query")
#     axes[0].axis('off')
    
#     axes[1].imshow(gt_mask, cmap='gray')
#     axes[1].set_title("GT")
#     axes[1].axis('off')
    
#     if pred_results.get('rough_mask') is not None:
#         axes[2].imshow(pred_results['rough_mask'], cmap='gray')
#         axes[2].set_title("Rough")
#     axes[2].axis('off')
    
#     if pred_results.get('final_mask') is not None:
#         axes[3].imshow(pred_results['final_mask'], cmap='gray')
#         if gt_mask is not None:
#             intersection = (pred_results['final_mask'] > 0) & (gt_mask > 0)
#             union = (pred_results['final_mask'] > 0) | (gt_mask > 0)
#             iou = intersection.sum() / (union.sum() + 1e-8)
#             axes[3].set_title(f"Final (IoU: {iou:.3f})")
#     axes[3].axis('off')
    
#     plt.suptitle(title)
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         plt.close()
#     else:
#         plt.show()


# if __name__ == '__main__':
#     print("=" * 60)
#     print("SP-SAM 模型多层特征版本")
#     print("=" * 60)
#     print(f"\n当前配置:")
#     print(f"  USE_MULTI_LAYER = {USE_MULTI_LAYER}")
#     print(f"  USE_MULTI_SCALE = {USE_MULTI_SCALE}")
#     print(f"  USE_DUAL_PROTOTYPE = {USE_DUAL_PROTOTYPE}")
#     print(f"  USE_DENSE_POINTS = {USE_DENSE_POINTS}")
#     if USE_MULTI_LAYER:
#         print(f"  DINO_LAYERS = {DINO_LAYERS}")
#         print(f"  LAYER_FUSION = {LAYER_FUSION}")
#         if LAYER_FUSION == 'weighted':
#             print(f"  LAYER_WEIGHTS = {LAYER_WEIGHTS}")
#     print("\n推荐实验:")
#     print("  1. USE_MULTI_LAYER=True, LAYER_FUSION='mean'")
#     print("  2. USE_MULTI_LAYER=True, LAYER_FUSION='weighted'")
#     print("  3. USE_MULTI_LAYER=True, DINO_LAYERS=[12, 18, 24]")
    





"""
SP-SAM Complete - 支持多种原型融合方式
=====================================

新增 PROTOTYPE_FUSION 配置：
- 'mean': 原型取平均后计算相似度（原方式）
- 'max': 每个原型单独算相似度，逐像素取最大值（新方式）
- 'mean_sim': 每个原型单独算相似度，逐像素取平均值
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Tuple, Optional, Any


# ============================================================
# 🔧 配置选项
# ============================================================
USE_MULTI_LAYER = False      # 模型多层特征
USE_MULTI_SCALE = True       # 图像多尺度（推荐，64.97%）
USE_DUAL_PROTOTYPE = False   # 前景+背景双原型
USE_DENSE_POINTS = False     # 更多点提示 (20正+10负)

# ⭐ 新增：原型融合方式
PROTOTYPE_FUSION = 'mean_sim'     # 'mean': 原型平均66.99%miou, 'max': 保留独立性取最大66.38%miou, 'mean_sim': 相似度平均66.83%miou

# 多层特征配置（如果USE_MULTI_LAYER=True）
DINO_LAYERS = [8, 16, 24]
LAYER_FUSION = 'weighted'
LAYER_WEIGHTS = [0.2, 0.3, 0.5]


class CMRSModule:
    """
    CMRS模块 - 支持多种原型融合方式
    
    PROTOTYPE_FUSION:
    - 'mean': K个原型 → 平均 → 1个原型 → 1个相似度图（原方式）
    - 'max': K个原型 → K个相似度图 → 逐像素取Max（新方式，保留独立性）
    - 'mean_sim': K个原型 → K个相似度图 → 逐像素取Mean
    """
    
    def __init__(self, dino_model, dino_transform, device='cuda'):
        self.dino_model = dino_model
        self.dino_transform = dino_transform
        self.device = device
        self.current_prototype = None
        self.current_bg_prototype = None
        self.current_query_features = None
        self.feature_channels = None
        
        # 保存所有原型（用于可视化/调试）
        self.all_prototypes = []
        self.all_similarity_maps = []
        
        # 打印配置
        config_str = []
        if USE_MULTI_LAYER:
            config_str.append(f"multi_layer(layers={DINO_LAYERS}, fusion={LAYER_FUSION})")
        if USE_MULTI_SCALE:
            config_str.append("multi_scale")
        if USE_DUAL_PROTOTYPE:
            config_str.append("dual_prototype")
        if USE_DENSE_POINTS:
            config_str.append("dense_points")
        config_str.append(f"proto_fusion={PROTOTYPE_FUSION}")
        print(f"📌 CMRS配置: {' + '.join(config_str)}")
    
    # ============================================================
    # 特征提取方法
    # ============================================================
    
    def extract_dino_features(self, img_pil):
        """标准单层特征提取（baseline）"""
        img_tensor = self.dino_transform(img_pil)[None].to(self.device)
        
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            features = self.dino_model.get_intermediate_layers(
                img_tensor.to(torch.bfloat16)
            )[0]
            h = w = int(features.shape[1] ** 0.5)
            feature_map = features.reshape(1, h, w, -1).permute(0, 3, 1, 2).float()
        
        return feature_map.squeeze(0)
    
    def extract_dino_features_multiscale(self, img_pil):
        """图像多尺度特征提取（推荐方案）"""
        scales = [0.75, 1.0, 1.25]
        original_size = img_pil.size
        
        all_features = []
        for scale in scales:
            new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            img_scaled = img_pil.resize(new_size, Image.BILINEAR)
            
            img_tensor = self.dino_transform(img_scaled)[None].to(self.device)
            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                features = self.dino_model.get_intermediate_layers(
                    img_tensor.to(torch.bfloat16)
                )[0]
                h = w = int(features.shape[1] ** 0.5)
                feature_map = features.reshape(1, h, w, -1).permute(0, 3, 1, 2).float()
            
            all_features.append(feature_map.squeeze(0))
        
        # 统一到中间尺度的大小
        target_h, target_w = all_features[1].shape[1], all_features[1].shape[2]
        fused_features = []
        for feat in all_features:
            if feat.shape[1] != target_h or feat.shape[2] != target_w:
                feat = F.interpolate(feat.unsqueeze(0), size=(target_h, target_w),
                                    mode='bilinear', align_corners=False).squeeze(0)
            fused_features.append(feat)
        
        fused = torch.stack(fused_features, dim=0).mean(dim=0)
        return fused
    
    def extract_dino_features_multilayer(self, img_pil):
        """模型多层特征提取"""
        img_tensor = self.dino_transform(img_pil)[None].to(self.device)
        
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            total_layers = len(self.dino_model.blocks)
            
            if total_layers >= 24:
                layers_to_extract = [7, 15, 23]
            elif total_layers >= 12:
                layers_to_extract = [3, 7, 11]
            elif total_layers >= 6:
                layers_to_extract = [1, 3, 5]
            else:
                layers_to_extract = [total_layers - 1]
            
            features_list = self.dino_model.get_intermediate_layers(
                img_tensor.to(torch.bfloat16),
                n=layers_to_extract,
                reshape=False
            )
            
            all_feature_maps = []
            for features in features_list:
                h = w = int(features.shape[1] ** 0.5)
                feature_map = features.reshape(1, h, w, -1).permute(0, 3, 1, 2).float()
                all_feature_maps.append(feature_map.squeeze(0))
            
            if len(all_feature_maps) == 1:
                fused = all_feature_maps[0]
            elif LAYER_FUSION == 'mean':
                fused = torch.stack(all_feature_maps, dim=0).mean(dim=0)
            elif LAYER_FUSION == 'concat':
                fused = torch.cat(all_feature_maps, dim=0)
            elif LAYER_FUSION == 'weighted':
                weights = LAYER_WEIGHTS[:len(all_feature_maps)]
                weights = [w / sum(weights) for w in weights]
                fused = sum(w * f for w, f in zip(weights, all_feature_maps))
            else:
                fused = torch.stack(all_feature_maps, dim=0).mean(dim=0)
        
        self.feature_channels = fused.shape[0]
        return fused
    
    def get_features(self, img_pil):
        """根据配置选择特征提取方式"""
        if USE_MULTI_LAYER:
            return self.extract_dino_features_multilayer(img_pil)
        elif USE_MULTI_SCALE:
            return self.extract_dino_features_multiscale(img_pil)
        else:
            return self.extract_dino_features(img_pil)
    
    # ============================================================
    # Mask处理
    # ============================================================
    
    def downsample_mask(self, mask_np, target_size):
        """下采样mask到特征图尺寸"""
        h, w = target_size
        resized_mask = cv2.resize(mask_np.astype(np.float32), (w, h),
                                  interpolation=cv2.INTER_NEAREST)
        resized_mask = (resized_mask > 0.5).astype(np.float32)
        return torch.from_numpy(resized_mask).to(self.device)
    
    # ============================================================
    # Prototype计算
    # ============================================================
    
    def compute_prototype(self, features, mask):
        """计算前景原型（Masked Average Pooling）"""
        mask_sum = mask.sum() + 1e-8
        prototype = (features * mask.unsqueeze(0)).sum(dim=(1, 2)) / mask_sum
        return prototype
    
    def compute_bg_prototype(self, features, mask):
        """计算背景原型"""
        bg_mask = 1.0 - mask
        bg_sum = bg_mask.sum() + 1e-8
        bg_prototype = (features * bg_mask.unsqueeze(0)).sum(dim=(1, 2)) / bg_sum
        return bg_prototype
    
    # ============================================================
    # ⭐ 相似度图计算（核心修改）
    # ============================================================
    
    def compute_similarity_map_prototype(self, query_features, support_features_list, support_masks_list):
        """
        计算query与support原型的相似度图
        
        PROTOTYPE_FUSION:
        - 'mean': K个原型取平均，再算相似度（原方式，信息压缩）
        - 'max': K个原型各自算相似度，逐像素取Max（新方式，保留独立性）
        - 'mean_sim': K个原型各自算相似度，逐像素取Mean
        """
        C, H, W = query_features.shape
        
        # 1. 计算每个support的前景原型
        fg_prototypes = []
        bg_prototypes = []
        
        for sup_feat, sup_mask in zip(support_features_list, support_masks_list):
            fg_proto = self.compute_prototype(sup_feat, sup_mask)
            fg_prototypes.append(fg_proto)
            
            if USE_DUAL_PROTOTYPE:
                bg_proto = self.compute_bg_prototype(sup_feat, sup_mask)
                bg_prototypes.append(bg_proto)
        
        # 保存所有原型（用于调试）
        self.all_prototypes = fg_prototypes
        
        # 2. Query特征预处理
        query_flat = query_features.reshape(C, -1).T      # (H*W, C)
        query_flat = F.normalize(query_flat, p=2, dim=1)  # L2归一化
        
        # 3. 根据融合策略计算相似度图
        if PROTOTYPE_FUSION == 'mean':
            # ====== 原方式：原型取平均后再算相似度 ======
            avg_fg_prototype = torch.stack(fg_prototypes, dim=0).mean(dim=0)
            self.current_prototype = F.normalize(avg_fg_prototype, p=2, dim=0)
            
            if USE_DUAL_PROTOTYPE and bg_prototypes:
                avg_bg_prototype = torch.stack(bg_prototypes, dim=0).mean(dim=0)
                self.current_bg_prototype = F.normalize(avg_bg_prototype, p=2, dim=0)
                
                fg_sim = torch.mv(query_flat, self.current_prototype)
                bg_sim = torch.mv(query_flat, self.current_bg_prototype)
                similarity_map = (fg_sim - bg_sim).reshape(H, W)
            else:
                similarity_scores = torch.mv(query_flat, self.current_prototype)
                similarity_map = similarity_scores.reshape(H, W)
            
            self.all_similarity_maps = [similarity_map]
        
        elif PROTOTYPE_FUSION == 'max':
            # ====== 新方式：每个原型单独算相似度，取最大值 ======
            # 保留每个Support的独立性，Query与每个Support比较，取最高匹配
            similarities = []
            
            for i, fg_proto in enumerate(fg_prototypes):
                proto_norm = F.normalize(fg_proto, p=2, dim=0)
                
                if USE_DUAL_PROTOTYPE and bg_prototypes:
                    bg_proto_norm = F.normalize(bg_prototypes[i], p=2, dim=0)
                    fg_sim = torch.mv(query_flat, proto_norm)
                    bg_sim = torch.mv(query_flat, bg_proto_norm)
                    sim = fg_sim - bg_sim
                else:
                    sim = torch.mv(query_flat, proto_norm)  # (H*W,)
                
                similarities.append(sim)
            
            # 逐像素取最大相似度（只要有一个Support匹配就行）
            stacked_sims = torch.stack(similarities, dim=0)  # (K, H*W)
            max_similarity, best_support_idx = stacked_sims.max(dim=0)  # (H*W,)
            similarity_map = max_similarity.reshape(H, W)
            
            # 保存所有相似度图和最佳匹配索引（用于分析）
            self.all_similarity_maps = [s.reshape(H, W) for s in similarities]
            self.best_support_map = best_support_idx.reshape(H, W)  # 每个像素最佳匹配的Support索引
            
            # 保存平均原型（用于兼容性）
            self.current_prototype = F.normalize(torch.stack(fg_prototypes, dim=0).mean(dim=0), p=2, dim=0)
        
        elif PROTOTYPE_FUSION == 'mean_sim':
            # ====== 每个原型单独算相似度，取平均 ======
            similarities = []
            
            for i, fg_proto in enumerate(fg_prototypes):
                proto_norm = F.normalize(fg_proto, p=2, dim=0)
                
                if USE_DUAL_PROTOTYPE and bg_prototypes:
                    bg_proto_norm = F.normalize(bg_prototypes[i], p=2, dim=0)
                    fg_sim = torch.mv(query_flat, proto_norm)
                    bg_sim = torch.mv(query_flat, bg_proto_norm)
                    sim = fg_sim - bg_sim
                else:
                    sim = torch.mv(query_flat, proto_norm)
                
                similarities.append(sim)
            
            # 逐像素取平均相似度
            stacked_sims = torch.stack(similarities, dim=0)  # (K, H*W)
            mean_similarity = stacked_sims.mean(dim=0)  # (H*W,)
            similarity_map = mean_similarity.reshape(H, W)
            
            self.all_similarity_maps = [s.reshape(H, W) for s in similarities]
            self.current_prototype = F.normalize(torch.stack(fg_prototypes, dim=0).mean(dim=0), p=2, dim=0)
        
        else:
            raise ValueError(f"Unknown PROTOTYPE_FUSION: {PROTOTYPE_FUSION}")
        
        self.current_query_features = query_features
        return similarity_map
    
    # ============================================================
    # 点提示采样
    # ============================================================
    
    def get_prompts_from_similarity(self, similarity_map, top_k=10, neg_k=5):
        """从相似度图采样正负点提示"""
        if USE_DENSE_POINTS:
            top_k = 20
            neg_k = 10
        
        sim_np = similarity_map.cpu().numpy()
        H, W = sim_np.shape
        mean_sim, std_sim, max_sim = sim_np.mean(), sim_np.std(), sim_np.max()
        
        # 阈值设置
        if USE_DUAL_PROTOTYPE:
            fg_threshold = max(0.1, mean_sim + 0.5 * std_sim)
            bg_threshold = min(-0.1, mean_sim - 0.5 * std_sim)
        else:
            fg_threshold = max(mean_sim + 1.0 * std_sim, max_sim * 0.5)
            bg_threshold = mean_sim - 0.5 * std_sim
        
        # 正样本采样
        fg_coords = np.argwhere(sim_np > fg_threshold)
        
        if len(fg_coords) < top_k:
            fg_threshold = mean_sim + 0.3 * std_sim if not USE_DUAL_PROTOTYPE else 0.0
            fg_coords = np.argwhere(sim_np > fg_threshold)
        
        if len(fg_coords) == 0:
            flat_indices = np.argsort(sim_np.flatten())[-top_k:]
            pos_points = np.array([np.unravel_index(idx, sim_np.shape) for idx in flat_indices])
        else:
            pos_points = self._sample_points_weighted_spatial(fg_coords, sim_np, top_k)
        
        # 负样本采样
        bg_coords = np.argwhere(sim_np < bg_threshold)
        
        if len(bg_coords) >= neg_k:
            neg_points = self._sample_negative_points_diverse(bg_coords, sim_np, neg_k, (H, W))
        else:
            neg_points = self._sample_from_borders(H, W, neg_k)
        
        # 合并
        all_points = np.vstack([pos_points, neg_points])
        all_labels = np.concatenate([np.ones(len(pos_points), dtype=np.int32),
                                     np.zeros(len(neg_points), dtype=np.int32)])
        return all_points[:, ::-1], all_labels
    
    def _sample_points_weighted_spatial(self, coords, sim_np, num_points):
        """加权空间分散采样正样本"""
        if len(coords) == 0:
            return np.array([[sim_np.shape[0]//2, sim_np.shape[1]//2]])
        
        sim_values = np.array([sim_np[y, x] for y, x in coords])
        selected, selected_indices = [], []
        
        # 第一个点：相似度最高
        first_idx = np.argmax(sim_values)
        selected.append(coords[first_idx])
        selected_indices.append(first_idx)
        
        # 后续点：平衡相似度和空间分散
        for _ in range(num_points - 1):
            if len(selected_indices) >= len(coords):
                break
            best_score, best_idx = -np.inf, -1
            for i, (coord, sim_val) in enumerate(zip(coords, sim_values)):
                if i in selected_indices:
                    continue
                min_dist = min(np.sqrt((coord[0]-s[0])**2 + (coord[1]-s[1])**2) for s in selected)
                score = sim_val * 0.7 + (min_dist / max(sim_np.shape)) * 0.3
                if score > best_score:
                    best_score, best_idx = score, i
            if best_idx >= 0:
                selected.append(coords[best_idx])
                selected_indices.append(best_idx)
        
        return np.array(selected)
    
    def _sample_negative_points_diverse(self, bg_coords, sim_np, num_points, shape):
        """3×3网格分散采样负样本"""
        H, W = shape
        grid_h, grid_w = 3, 3
        cell_h, cell_w = H // grid_h, W // grid_w
        neg_points = []
        
        for gh in range(grid_h):
            for gw in range(grid_w):
                if len(neg_points) >= num_points:
                    break
                y_start, y_end = gh * cell_h, (gh + 1) * cell_h
                x_start, x_end = gw * cell_w, (gw + 1) * cell_w
                cell_mask = ((bg_coords[:, 0] >= y_start) & (bg_coords[:, 0] < y_end) &
                            (bg_coords[:, 1] >= x_start) & (bg_coords[:, 1] < x_end))
                cell_coords = bg_coords[cell_mask]
                if len(cell_coords) > 0:
                    cell_sims = np.array([sim_np[y, x] for y, x in cell_coords])
                    neg_points.append(cell_coords[np.argmin(cell_sims)])
        
        if len(neg_points) < num_points and len(bg_coords) > 0:
            indices = np.random.choice(len(bg_coords), 
                                      min(num_points - len(neg_points), len(bg_coords)), 
                                      replace=False)
            for idx in indices:
                neg_points.append(bg_coords[idx])
        
        return np.array(neg_points[:num_points]) if neg_points else np.array([[0, 0]])
    
    def _sample_from_borders(self, H, W, num_points):
        """从边界采样负样本（保底策略）"""
        border = 3
        corners = [[border, border], [border, W-border-1], 
                   [H-border-1, border], [H-border-1, W-border-1]]
        neg_points = corners[:min(num_points, 4)]
        
        while len(neg_points) < num_points:
            side = np.random.randint(4)
            if side == 0:
                neg_points.append([border, np.random.randint(border, W-border)])
            elif side == 1:
                neg_points.append([H-border-1, np.random.randint(border, W-border)])
            elif side == 2:
                neg_points.append([np.random.randint(border, H-border), border])
            else:
                neg_points.append([np.random.randint(border, H-border), W-border-1])
        
        return np.array(neg_points)
    
    # ============================================================
    # 主函数：生成粗分割
    # ============================================================
    
    def generate_rough_mask(self, query_img, support_images, support_masks, 
                           sam2_predictor, top_k=10, neg_k=5):
        """生成粗分割掩码"""
        # 1. 提取query特征
        query_features = self.get_features(query_img)
        C, H_feat, W_feat = query_features.shape
        
        # 2. 提取所有support的特征和下采样mask
        support_features_list, support_masks_list = [], []
        for sup_img, sup_mask in zip(support_images, support_masks):
            sup_features = self.get_features(sup_img)
            downsampled_mask = self.downsample_mask(sup_mask, 
                                                    (sup_features.shape[1], sup_features.shape[2]))
            support_features_list.append(sup_features)
            support_masks_list.append(downsampled_mask)
        
        # 3. 计算相似度图
        similarity_map = self.compute_similarity_map_prototype(
            query_features, support_features_list, support_masks_list
        )
        
        # 4. 采样点提示
        point_coords_feat, point_labels = self.get_prompts_from_similarity(
            similarity_map, top_k, neg_k
        )
        
        # 5. 坐标缩放：特征图尺度 → 原图尺度
        H_img, W_img = query_img.size[::-1]
        H_feat, W_feat = similarity_map.shape
        scale_y, scale_x = H_img / H_feat, W_img / W_feat
        
        point_coords_img = point_coords_feat.copy().astype(np.float32)
        point_coords_img[:, 0] *= scale_x
        point_coords_img[:, 1] *= scale_y
        point_coords_img = point_coords_img.astype(np.int32)
        
        # 6. SAM2预测
        sam2_predictor.set_image(np.array(query_img))
        
        try:
            masks, scores, _ = sam2_predictor.predict(
                point_coords=point_coords_img,
                point_labels=point_labels,
                multimask_output=True
            )
            rough_mask = masks[np.argmax(scores)]
        except Exception as e:
            print(f"   ⚠️  SAM2预测失败: {e}")
            rough_mask = None
        
        return rough_mask, similarity_map.cpu().numpy(), point_coords_img, point_labels


class SAM2MemoryModule:
    """Memory模块 - 用于精炼分割结果"""
    
    def __init__(self, sam2_model, device='cuda', sam2_model_type='large'):
        self.model = sam2_model
        self.device = device
        self.sam2_model_type = sam2_model_type
        self.prev_out = {}
        self.support_set = False
        self.image_size = getattr(sam2_model, 'image_size', 1024)
        self.num_feature_levels = getattr(sam2_model, 'num_feature_levels', 3)
    
    def clear_support(self):
        self.prev_out = {}
        self.support_set = False
    
    def _prepare_image(self, img):
        target_size = self.image_size
        img_resized = img.resize((target_size, target_size), Image.BILINEAR)
        img_np = np.array(img_resized).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        return img_tensor.unsqueeze(0).to(self.device)
    
    def _prepare_mask(self, mask, target_size=None):
        if target_size is None:
            target_size = self.image_size
        mask_resized = cv2.resize(mask.astype(np.float32), (target_size, target_size),
                                  interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mask_resized).float()
        return mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
    
    def encode_support(self, support_img, support_mask):
        """编码support样本到memory"""
        img_tensor = self._prepare_image(support_img)
        
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            backbone_out = self.model.forward_image(img_tensor)
            feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
            
            for i in range(len(feature_maps)):
                feature_maps[i] = feature_maps[i].clone()
            
            pix_feat = feature_maps[-1]
            mask_tensor = self._prepare_mask(support_mask)
            high_res_masks = mask_tensor.float() * 20.0 - 10.0
            
            maskmem_out = self.model.memory_encoder(
                pix_feat=pix_feat, masks=high_res_masks, skip_mask_sigmoid=True
            )
            
            self.prev_out.setdefault("maskmem_features", []).append(
                maskmem_out["vision_features"].clone()
            )
            if "maskmem_pos_enc" not in self.prev_out:
                self.prev_out["maskmem_pos_enc"] = [
                    m.clone() for m in maskmem_out["vision_pos_enc"]
                ]
        
        self.support_set = True
    
    def set_support(self, support_images, support_masks, target_size=None):
        """设置support集合"""
        self.clear_support()
        for img, mask in zip(support_images, support_masks):
            self.encode_support(img, mask)
        return self.support_set
    
    def predict_query(self, query_img, rough_mask=None, point_coords=None, point_labels=None):
        """预测query图像的分割（使用Memory Attention）"""
        if not self.support_set:
            raise ValueError("请先调用set_support()")
        
        original_size = query_img.size[::-1]
        img_tensor = self._prepare_image(query_img)
        B = img_tensor.shape[0]
        
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            backbone_out = self.model.forward_image(img_tensor)
            feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
            vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels:]
            
            for i in range(len(feature_maps)):
                feature_maps[i] = feature_maps[i].clone()
                vision_pos_embeds[i] = vision_pos_embeds[i].clone()
            
            high_res_features = feature_maps[:-1]
            pix_feat = feature_maps[-1]
            
            # Memory Attention
            to_cat_memory = [m.flatten(2).permute(2, 0, 1) 
                            for m in self.prev_out["maskmem_features"]]
            to_cat_memory_pos_embed = len(self.prev_out["maskmem_features"]) * [
                self.prev_out["maskmem_pos_enc"][0].flatten(2).permute(2, 0, 1)
            ]
            
            memory = torch.cat(to_cat_memory, dim=0)
            memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)
            
            pix_feat_with_mem = self.model.memory_attention(
                curr=pix_feat.flatten(2).permute(2, 0, 1),
                curr_pos=vision_pos_embeds[-1].flatten(2).permute(2, 0, 1),
                memory=memory, memory_pos=memory_pos_embed, num_obj_ptr_tokens=0
            )
            pix_feat_with_mem = pix_feat_with_mem.clone().permute(1, 2, 0).view(*pix_feat.shape)
            
            # 点提示
            if point_coords is not None and point_labels is not None:
                scale_x = self.image_size / original_size[1]
                scale_y = self.image_size / original_size[0]
                
                scaled_coords = point_coords.copy().astype(np.float32)
                scaled_coords[:, 0] *= scale_x
                scaled_coords[:, 1] *= scale_y
                
                sam_point_coords = torch.from_numpy(scaled_coords).unsqueeze(0).float().to(self.device)
                sam_point_labels = torch.from_numpy(point_labels.astype(np.int32)).unsqueeze(0).to(self.device)
            else:
                sam_point_coords = torch.zeros(B, 1, 2, device=self.device)
                sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=self.device)
            
            # Mask Prompt
            sam_mask_prompt = None
            if rough_mask is not None:
                mask_tensor = self._prepare_mask(rough_mask, target_size=self.image_size)
                high_res_mask_logits = mask_tensor.float() * 20.0 - 10.0
                sam_mask_prompt = F.interpolate(high_res_mask_logits, size=(256, 256),
                                                mode='bilinear', align_corners=False)
            
            sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
                points=(sam_point_coords, sam_point_labels), boxes=None, masks=sam_mask_prompt
            )
            sparse_embeddings = sparse_embeddings.clone()
            dense_embeddings = dense_embeddings.clone()
            image_pe = self.model.sam_prompt_encoder.get_dense_pe().clone()
            
            low_res_masks, ious, _, _ = self.model.sam_mask_decoder(
                image_embeddings=pix_feat_with_mem, image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=False, high_res_features=high_res_features
            )
            
            best_idx = torch.argmax(ious[0])
            best_mask = low_res_masks[0, best_idx:best_idx+1].unsqueeze(0)
            best_iou = ious[0, best_idx]
            
            best_mask = best_mask.float()
            high_res_masks = F.interpolate(best_mask, size=(self.image_size, self.image_size),
                                           mode='bilinear', align_corners=False)
            final_masks = F.interpolate(high_res_masks, size=original_size,
                                        mode='bilinear', align_corners=False)
            
            pred_mask = (final_masks[0, 0] > 0).cpu().numpy().astype(np.uint8)
            score = float(best_iou.cpu())
        
        return pred_mask, score
    
    def predict_with_mask_prompt_only(self, query_img, rough_mask, point_coords=None, point_labels=None):
        """
        ★ 实验2专用：只使用Mask Decoder + rough mask prompt，不使用Memory Attention
        
        这是cmrs_predictor模式的正确实现：
        - 使用SAM2的Image Encoder提取特征
        - 使用rough_mask作为mask prompt
        - 直接送入Mask Decoder（不经过Memory Attention）
        
        Args:
            query_img: 查询图像 (PIL.Image)
            rough_mask: CMRS生成的粗分割mask
            point_coords: 点坐标（可选）
            point_labels: 点标签（可选）
            
        Returns:
            pred_mask: 预测mask
            score: 预测得分
        """
        original_size = query_img.size[::-1]  # (H, W)
        img_tensor = self._prepare_image(query_img)
        B = img_tensor.shape[0]
        
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            # 1. Image Encoder
            backbone_out = self.model.forward_image(img_tensor)
            feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
            vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels:]
            
            for i in range(len(feature_maps)):
                feature_maps[i] = feature_maps[i].clone()
                vision_pos_embeds[i] = vision_pos_embeds[i].clone()
            
            high_res_features = feature_maps[:-1]
            pix_feat = feature_maps[-1]
            
            # ★ 注意：这里不使用Memory Attention，直接用pix_feat
            # pix_feat_with_mem = pix_feat  # 不经过Memory Attention
            
            # 2. 点提示（可选）
            if point_coords is not None and point_labels is not None:
                scale_x = self.image_size / original_size[1]
                scale_y = self.image_size / original_size[0]
                
                scaled_coords = point_coords.copy().astype(np.float32)
                scaled_coords[:, 0] *= scale_x
                scaled_coords[:, 1] *= scale_y
                
                sam_point_coords = torch.from_numpy(scaled_coords).unsqueeze(0).float().to(self.device)
                sam_point_labels = torch.from_numpy(point_labels.astype(np.int32)).unsqueeze(0).to(self.device)
            else:
                sam_point_coords = torch.zeros(B, 1, 2, device=self.device)
                sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=self.device)
            
            # 3. Mask Prompt（核心！）
            sam_mask_prompt = None
            if rough_mask is not None:
                mask_tensor = self._prepare_mask(rough_mask, target_size=self.image_size)
                high_res_mask_logits = mask_tensor.float() * 20.0 - 10.0
                sam_mask_prompt = F.interpolate(high_res_mask_logits, size=(256, 256),
                                                mode='bilinear', align_corners=False)
            
            # 4. Prompt Encoder
            sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
                points=(sam_point_coords, sam_point_labels), boxes=None, masks=sam_mask_prompt
            )
            sparse_embeddings = sparse_embeddings.clone()
            dense_embeddings = dense_embeddings.clone()
            image_pe = self.model.sam_prompt_encoder.get_dense_pe().clone()
            
            # 5. Mask Decoder（★ 使用pix_feat而不是pix_feat_with_mem）
            low_res_masks, ious, _, _ = self.model.sam_mask_decoder(
                image_embeddings=pix_feat,  # ★ 不使用Memory增强的特征
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=False, 
                high_res_features=high_res_features
            )
            
            # 6. 选择最佳mask
            best_idx = torch.argmax(ious[0])
            best_mask = low_res_masks[0, best_idx:best_idx+1].unsqueeze(0)
            best_iou = ious[0, best_idx]
            
            best_mask = best_mask.float()
            high_res_masks = F.interpolate(best_mask, size=(self.image_size, self.image_size),
                                           mode='bilinear', align_corners=False)
            final_masks = F.interpolate(high_res_masks, size=original_size,
                                        mode='bilinear', align_corners=False)
            
            pred_mask = (final_masks[0, 0] > 0).cpu().numpy().astype(np.uint8)
            score = float(best_iou.cpu())
        
        return pred_mask, score


class SPSAMModel:
    """SP-SAM主模型"""
    
    def __init__(self, sam2_model, sam2_predictor, dino_model, dino_transform,
                 device='cuda', sam2_model_type='large'):
        self.sam2_model = sam2_model
        self.sam2_predictor = sam2_predictor
        self.device = device
        
        self.cmrs = CMRSModule(dino_model, dino_transform, device)
        self.memory_module = SAM2MemoryModule(sam2_model, device, sam2_model_type)
        
        self._support_images = []
        self._support_masks = []
        self._support_set = False
        
        print(f"✅ SPSAMModel 初始化完成")
    
    def set_support(self, support_images, support_masks):
        """设置support集合"""
        self._support_images = support_images
        self._support_masks = support_masks
        success = self.memory_module.set_support(support_images, support_masks)
        self._support_set = success
        return success
    
    def clear_support(self):
        """清除support集合"""
        self._support_images = []
        self._support_masks = []
        self._support_set = False
        self.memory_module.clear_support()
    
    def predict(self, query_img, support_images, support_masks, 
                use_cmrs=True, use_memory_refinement=False, use_predictor_refinement=False):
        """预测接口"""
        if use_memory_refinement:
            self.set_support(support_images, support_masks)
        return self.predict_query(query_img, use_cmrs, use_memory_refinement,
                                  use_predictor_refinement,  # 传递新参数
                                  support_images, support_masks)
    
    def predict_query(self, query_img, use_cmrs=True, use_memory_refinement=False,
                     use_predictor_refinement=False,  # ★ 新增参数：是否使用SAM2 Predictor精炼
                     support_images=None, support_masks=None):
        """
        预测query图像的分割
        
        Args:
            query_img: 查询图像 (PIL.Image)
            use_cmrs: 是否使用CMRS生成rough mask
            use_memory_refinement: 是否使用Memory机制精炼
            use_predictor_refinement: 是否使用SAM2 Predictor精炼（用于cmrs_predictor模式）
            support_images: support图像列表
            support_masks: support mask列表
            
        模式对应：
            - rough_only: use_cmrs=True, use_memory_refinement=False, use_predictor_refinement=False
            - cmrs_predictor: use_cmrs=True, use_memory_refinement=False, use_predictor_refinement=True
            - memory_only: use_cmrs=False, use_memory_refinement=True
            - cmrs_memory: use_cmrs=True, use_memory_refinement=True
        """
        sup_imgs = support_images if support_images else self._support_images
        sup_masks = support_masks if support_masks else self._support_masks
        
        if len(sup_imgs) == 0:
            raise ValueError("No support samples")
        
        results = {'final_mask': None, 'rough_mask': None, 
                  'similarity_map': None, 'score': 0.0}
        
        rough_mask = None
        point_coords = None
        point_labels = None
        
        # Stage 1: CMRS生成粗分割
        if use_cmrs:
            rough_mask, similarity_map, point_coords, point_labels = \
                self.cmrs.generate_rough_mask(
                    query_img, sup_imgs, sup_masks, self.sam2_predictor
                )
            results['rough_mask'] = rough_mask
            results['similarity_map'] = similarity_map
            
            # 保存所有Support的相似度图（用于分析）
            if hasattr(self.cmrs, 'all_similarity_maps'):
                results['all_similarity_maps'] = [s.cpu().numpy() if torch.is_tensor(s) else s 
                                                   for s in self.cmrs.all_similarity_maps]
        
        # Stage 2: 选择精炼方式
        if use_memory_refinement:
            # Memory模块精炼（cmrs_memory 或 memory_only 模式）
            if not self._support_set:
                self.set_support(sup_imgs, sup_masks)
            
            pred_mask, score = self.memory_module.predict_query(
                query_img,
                rough_mask=rough_mask if use_cmrs else None,
                point_coords=point_coords,
                point_labels=point_labels
            )
            results['final_mask'] = pred_mask
            results['score'] = score
            
        elif use_predictor_refinement and rough_mask is not None:
            # ★ SAM2 Predictor精炼（cmrs_predictor模式）
            # 使用rough_mask作为mask prompt，再做一次SAM2预测
            pred_mask, score = self._refine_with_predictor(
                query_img, rough_mask, point_coords, point_labels
            )
            results['final_mask'] = pred_mask
            results['score'] = score
            
        else:
            # 不做精炼，直接返回rough_mask（rough_only模式）
            if rough_mask is not None:
                results['final_mask'] = rough_mask.astype(np.uint8)
                results['score'] = 1.0
            else:
                H, W = query_img.size[::-1]
                results['final_mask'] = np.zeros((H, W), dtype=np.uint8)
                results['score'] = 0.0
        
        return results
    
    def _refine_with_predictor(self, query_img, rough_mask, point_coords=None, point_labels=None):
        """
        使用Mask Decoder对rough_mask进行精炼（不使用Memory）
        
        这是cmrs_predictor模式的正确实现：
        - Image Encoder → Mask Decoder（跳过Memory Attention）
        - rough_mask作为mask prompt
        
        Args:
            query_img: 查询图像 (PIL.Image)
            rough_mask: CMRS生成的粗分割mask
            point_coords: 点坐标（可选）
            point_labels: 点标签（可选）
            
        Returns:
            refined_mask: 精炼后的mask
            score: 预测得分
        """
        try:
            # 调用Memory模块的新方法（不使用Memory Attention）
            pred_mask, score = self.memory_module.predict_with_mask_prompt_only(
                query_img, rough_mask, point_coords, point_labels
            )
            return pred_mask, score
            
        except Exception as e:
            print(f"   ⚠️  Mask Decoder精炼失败: {e}")
            import traceback
            traceback.print_exc()
            # 失败时返回原始rough_mask
            return rough_mask.astype(np.uint8), 0.5


def visualize_sp_sam_results(query_img, gt_mask, pred_results,
                            support_images=None, support_masks=None,
                            save_path=None, title="SP-SAM Results"):
    """可视化结果"""
    import matplotlib.pyplot as plt
    
    n_cols = 4
    fig, axes = plt.subplots(1, n_cols, figsize=(16, 4))
    
    axes[0].imshow(query_img)
    axes[0].set_title("Query")
    axes[0].axis('off')
    
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title("GT")
    axes[1].axis('off')
    
    if pred_results.get('rough_mask') is not None:
        axes[2].imshow(pred_results['rough_mask'], cmap='gray')
        axes[2].set_title("Rough")
    axes[2].axis('off')
    
    if pred_results.get('final_mask') is not None:
        axes[3].imshow(pred_results['final_mask'], cmap='gray')
        if gt_mask is not None:
            intersection = (pred_results['final_mask'] > 0) & (gt_mask > 0)
            union = (pred_results['final_mask'] > 0) | (gt_mask > 0)
            iou = intersection.sum() / (union.sum() + 1e-8)
            axes[3].set_title(f"Final (IoU: {iou:.3f})")
    axes[3].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    print("=" * 60)
    print("SP-SAM 支持多种原型融合方式")
    print("=" * 60)
    print(f"\n当前配置:")
    print(f"  USE_MULTI_SCALE = {USE_MULTI_SCALE}")
    print(f"  USE_MULTI_LAYER = {USE_MULTI_LAYER}")
    print(f"  USE_DUAL_PROTOTYPE = {USE_DUAL_PROTOTYPE}")
    print(f"  USE_DENSE_POINTS = {USE_DENSE_POINTS}")
    print(f"  PROTOTYPE_FUSION = {PROTOTYPE_FUSION}")
    print("\n原型融合方式说明:")
    print("  'mean': K个原型取平均，信息压缩为1个原型")
    print("  'max': K个原型各自算相似度，逐像素取Max（推荐，保留独立性）")
    print("  'mean_sim': K个原型各自算相似度，逐像素取Mean")