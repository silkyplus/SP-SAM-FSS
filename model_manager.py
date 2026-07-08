from src.libs import *
from src.helper_functions import *


class ModelManager:
    def __init__(self, device='cpu'):
        self.device = device
        self.dinov2_model = None
        self.dinov3_model = None
        self.sam2_model = None
        self.sam2_predictor = None
        self.sam2_mask_generator = None

    def load_dinov2_model(self, dinov2_model_name='dinov2_vitl14', weights_path=None, dinov2_path='facebookresearch_dinov2_main'):
        """
        加载DINOv2模型
        
        Args:
            dinov2_model_name: 模型名称 (dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14)
            weights_path: 本地权重路径（可选）
            dinov2_path: DINOv2项目路径
        """
        if self.dinov2_model is None:
            try:
                if weights_path:
                    print(f"📦 从本地加载 DINOv2: {dinov2_model_name}")
                    print(f"   权重路径: {weights_path}")
                    self.dinov2_model = torch.hub.load(
                        repo_or_dir=dinov2_path,
                        model=dinov2_model_name,
                        source='local',
                        pretrained=False
                    )
                    state_dict = torch.load(weights_path, map_location='cpu')
                    if 'model' in state_dict:
                        state_dict = state_dict['model']
                    self.dinov2_model.load_state_dict(state_dict, strict=False)
                    print("   ✅ 本地权重加载成功")
                else:
                    print(f"📦 从 torch.hub 加载 DINOv2: {dinov2_model_name}")
                    self.dinov2_model = torch.hub.load(
                        repo_or_dir="facebookresearch/dinov2", 
                        model=dinov2_model_name
                    )
                
                self.dinov2_model = self.dinov2_model.to(torch.bfloat16)
                self.dinov2_model.eval()
                self.dinov2_model.to(self.device)
                print("✅ DINOv2 模型加载完成\n")
                
            except Exception as e:
                print(f"❌ DINOv2 加载失败: {e}")
                raise

        self.dinov2_transform = transforms.Compose([
            transforms.Resize(size=(518, 518), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        return self.dinov2_model, self.dinov2_transform
    #D:\vscode\python_project\dinov3-main-\dinov3-main\weights\dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
    #D:\pycharm_projects\NOTRAING\NOTRAING\dinov3_weights\dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
    def load_dinov3_model(self, dinov3_model_name='dinov3_vitb16', weights_path=r'D:\vscode\python_project\dinov3-main-\dinov3-main\weights\dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth', dinov3_path='dinov3_main'):
        """
        加载DINOv3模型
        
        Args:
            dinov3_model_name: 模型名称
            weights_path: 本地权重路径（可选）
            dinov3_path: DINOv3项目路径
        """
        if self.dinov3_model is None:
            is_7b_model = weights_path and '7b' in weights_path.lower()
            
            try:
                if not is_7b_model:
                    print(f"📦 从本地加载 DINOv3: {dinov3_model_name}")
                    print(f"   项目路径: {dinov3_path}")
                    self.dinov3_model = torch.hub.load(
                        repo_or_dir=dinov3_path,
                        model=dinov3_model_name,
                        source='local',
                        pretrained=(weights_path is None)
                    ).to(torch.bfloat16)
                    
                    if weights_path is not None:
                        print(f"   权重路径: {weights_path}")
                        state_dict = torch.load(weights_path, map_location='cpu')
                        if 'model' in state_dict:
                            state_dict = state_dict['model']
                        self.dinov3_model.load_state_dict(state_dict, strict=False)
                        print("   ✅ 本地权重加载成功")
                else:
                    import sys
                    sys.path.insert(0, dinov3_path)
                    from dinov3.models import vision_transformer as vits
                    
                    print("📦 加载 7B 模型架构...")
                    self.dinov3_model = vits.DinoVisionTransformer(
                        img_size=560,
                        patch_size=16,
                        embed_dim=4096,
                        depth=24,
                        num_heads=32,
                    )
                    
                    if weights_path:
                        print(f"   权重路径: {weights_path}")
                        state_dict = torch.load(weights_path, map_location='cpu')
                        if 'model' in state_dict:
                            state_dict = state_dict['model']
                        self.dinov3_model.load_state_dict(state_dict, strict=False)
                        del state_dict
                        import gc
                        gc.collect()
                    
                    self.dinov3_model = self.dinov3_model.to(torch.bfloat16)
                
                self.dinov3_model.eval()
                self.dinov3_model.to(self.device)
                print("✅ DINOv3 模型加载完成\n")
                
            except Exception as e:
                print(f"❌ DINOv3 加载失败: {e}")
                raise

        input_size = 560 if 'vitl16' in dinov3_model_name or is_7b_model else 518
            
        self.dinov3_transform = transforms.Compose([
            transforms.Resize(size=(input_size, input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC, 
                            antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        return self.dinov3_model, self.dinov3_transform

    def load_sam2_model(self, sam2_model_type='tiny', model_cfg=None, sam2_model_ckpt_path=None, 
                       points_per_side=16, mode='whole_object'):
        """
        加载SAM2模型
        
        Args:
            sam2_model_type: 模型类型
            model_cfg: 配置文件路径
            sam2_model_ckpt_path: 权重路径
            points_per_side: 采样点密度（越小越倾向整体，推荐 16-24）
            mode: 'whole_object'（整体分割）或 'detailed'（细节分割）
        """
        if self.sam2_model is None:
            if sam2_model_ckpt_path is None:
                model_cfg, ckpt_path = get_sam2_model_cfg_and_ckpt_path(model_type=sam2_model_type)
            else:
                assert model_cfg is not None, 'Please provide model_cfg'
                model_cfg = model_cfg
                ckpt_path = sam2_model_ckpt_path
            
            self.sam2_model = build_sam2(model_cfg, ckpt_path, device=self.device, apply_postprocessing=False).to(self.device)
            self.sam2_predictor = SAM2ImagePredictor(sam_model=self.sam2_model)
            
            # 根据模式选择参数
            if mode == 'whole_object':
                # 整体分割模式：更少的采样点，更高的阈值，更大的最小面积
                print("🎯 SAM2 模式: 整体分割")
                self.sam2_mask_generator = SAM2AutomaticMaskGenerator(
                    model=self.sam2_model,
                    points_per_side=points_per_side,  # 降低采样密度（默认 16）
                    points_per_batch=128,
                    pred_iou_thresh=0.88,              # 提高 IoU 阈值（更准确的 mask）
                    stability_score_thresh=0.95,       # 提高稳定性阈值
                    stability_score_offset=0.7,
                    crop_n_layers=0,                   # 减少裁剪层（避免过细分割）
                    box_nms_thresh=0.7,
                    crop_n_points_downscale_factor=2,
                    min_mask_region_area=1000.0,       # 过滤小区域（至少 1000 像素）
                    use_m2m=True,
                )
            else:
                # 细节分割模式：更多采样点，更低阈值
                print("🔍 SAM2 模式: 细节分割")
                self.sam2_mask_generator = SAM2AutomaticMaskGenerator(
                    model=self.sam2_model,
                    points_per_side=32,                # 更密集的采样
                    points_per_batch=128,
                    pred_iou_thresh=0.7,
                    stability_score_thresh=0.92,
                    stability_score_offset=0.7,
                    crop_n_layers=1,
                    box_nms_thresh=0.7,
                    crop_n_points_downscale_factor=2,
                    min_mask_region_area=100.0,
                    use_m2m=True,
                )
                
        return self.sam2_model, self.sam2_predictor, self.sam2_mask_generator