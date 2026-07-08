"""
Pascal-5i 批量评估脚本（新格式）
================================

按顺序自动运行4组消融实验：
1. rough_only: 纯CMRS rough mask (基线)
2. cmrs_predictor: CMRS + SAM2 Predictor
3. memory_only: 纯Memory机制
4. cmrs_memory: CMRS + Memory (完整SP-SAM)

使用方法：
    # 评估所有folds的所有消融实验 (1-shot)
    python batch_evaluate_new.py --data_root pascal5i_output --k_shot 1
    
    # 评估所有folds的所有消融实验 (5-shot)
    python batch_evaluate_new.py --data_root pascal5i_output --k_shot 5
    
    # 快速测试：只评估fold 0
    python batch_evaluate_new.py --data_root pascal5i_output --k_shot 1 --fold 0
    
    # 限制每个类别的query数量（快速测试）
    python batch_evaluate_new.py --data_root pascal5i_output --k_shot 1 --max_query 10
"""

import argparse
import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime


def run_single_evaluation(data_root: str, k_shot: int, mode: str,
                         fold: int = None,
                         dino_model: str = 'dinov3_vitb16',
                         sam2_model: str = 'large',
                         device: str = 'cuda',
                         random_seed: int = 42,
                         max_query: int = None,
                         sorted_support_dir: str = None) -> tuple:
    """运行单次评估"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if fold is not None:
        output_dir = f'results_fold{fold}_{k_shot}shot_{mode}_{timestamp}'
    else:
        output_dir = f'results_all_{k_shot}shot_{mode}_{timestamp}'
    
    print("\n" + "="*80)
    print(f"运行评估: {k_shot}-shot, 模式: {mode}")
    if fold is not None:
        print(f"评估Fold: {fold}")
    else:
        print(f"评估所有Folds")
    print(f"输出目录: {output_dir}")
    print("="*80)
    
    # 构建命令
    cmd = [
        sys.executable, 'evaluate_pascal5i_new.py',
        '--data_root', data_root,
        '--k_shot', str(k_shot),
        '--mode', mode,
        '--output_dir', output_dir,
        '--dino_model', dino_model,
        '--sam2_model', sam2_model,
        '--device', device,
        '--random_seed', str(random_seed)
    ]
    
    if fold is not None:
        cmd.extend(['--fold', str(fold)])
    
    if max_query is not None:
        cmd.extend(['--max_query', str(max_query)])
    
    if sorted_support_dir:
        cmd.extend(['--sorted_support_dir', sorted_support_dir])
    
    print(f"\n命令: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ 评估完成: {output_dir}")
        return output_dir, True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 评估失败: {e}")
        return output_dir, False
    except FileNotFoundError as e:
        print(f"\n❌ 文件未找到: {e}")
        return output_dir, False


def batch_evaluate(data_root: str, k_shot: int = 1,
                  fold: int = None,
                  dino_model: str = 'dinov3_vitb16',
                  sam2_model: str = 'large',
                  device: str = 'cuda',
                  random_seed: int = 42,
                  max_query: int = None,
                  sorted_support_dir: str = None,
                  modes: list = None):
    """批量运行消融实验"""
    
    # 默认的消融实验顺序
    if modes is None:
        modes = [
            ('rough_only', '纯CMRS rough mask (基线)'),
            ('cmrs_predictor', 'CMRS + SAM2 Predictor'),
            ('memory_only', '纯Memory机制'),
            ('cmrs_memory', 'CMRS + Memory (完整SP-SAM)')
        ]
    
    print("\n" + "="*80)
    print("Pascal-5i 消融实验（新格式）")
    print("="*80)
    print(f"数据集路径: {data_root}")
    print(f"K-shot: {k_shot}")
    if fold is not None:
        print(f"评估Fold: {fold}")
    else:
        print(f"评估所有Folds: 0, 1, 2, 3")
    print(f"DINO模型: {dino_model}")
    print(f"SAM2模型: {sam2_model}")
    if max_query:
        print(f"每类最大Query: {max_query}")
    print(f"\n消融实验顺序:")
    for i, (mode, desc) in enumerate(modes, 1):
        print(f"  实验{i}: {mode} - {desc}")
    print("="*80)
    
    results = []
    
    for idx, (mode, desc) in enumerate(modes, 1):
        print(f"\n{'='*80}")
        print(f"实验 {idx}/{len(modes)}: {desc}")
        print(f"{'='*80}")
        
        output_dir, success = run_single_evaluation(
            data_root, k_shot, mode,
            fold=fold,
            dino_model=dino_model,
            sam2_model=sam2_model,
            device=device,
            random_seed=random_seed,
            max_query=max_query,
            sorted_support_dir=sorted_support_dir
        )
        
        results.append({
            'experiment': idx,
            'mode': mode,
            'description': desc,
            'output_dir': output_dir,
            'success': success
        })
    
    # 打印摘要
    print("\n" + "="*80)
    print("消融实验摘要")
    print("="*80)
    
    comparison_table = []
    
    for r in results:
        status = "✅ 成功" if r['success'] else "❌ 失败"
        print(f"\n实验{r['experiment']}: {r['description']}")
        print(f"   状态: {status}")
        print(f"   输出: {r['output_dir']}")
        
        if r['success']:
            # 尝试读取结果
            output_dir = Path(r['output_dir'])
            json_files = list(output_dir.glob('*.json'))
            
            if json_files:
                try:
                    with open(json_files[0], 'r') as f:
                        data = json.load(f)
                    
                    if 'overall_mean_iou' in data:
                        miou = data['overall_mean_iou']
                        comparison_table.append({
                            'experiment': r['experiment'],
                            'mode': r['mode'],
                            'description': r['description'],
                            'miou': miou
                        })
                        print(f"   Overall mIoU: {miou*100:.2f}%")
                    elif 'mean_iou' in data:
                        miou = data['mean_iou']
                        comparison_table.append({
                            'experiment': r['experiment'],
                            'mode': r['mode'],
                            'description': r['description'],
                            'miou': miou
                        })
                        print(f"   mIoU: {miou*100:.2f}%")
                except Exception as e:
                    print(f"   ⚠️ 无法读取结果: {e}")
    
    # 打印对比表格
    if comparison_table:
        print("\n" + "="*80)
        print("消融实验对比表")
        print("="*80)
        print(f"{'实验':<8} {'模式':<18} {'mIoU':<10} {'说明':<40}")
        print("-" * 80)
        for item in comparison_table:
            print(f"{item['experiment']:<8} {item['mode']:<18} {item['miou']*100:.2f}%     {item['description']}")
        
        # 计算改进幅度
        if len(comparison_table) > 1:
            baseline_miou = comparison_table[0]['miou']
            print(f"\n相对于基线 (实验1: {comparison_table[0]['mode']}) 的改进:")
            for item in comparison_table[1:]:
                improvement = item['miou'] - baseline_miou
                improvement_pct = (improvement / baseline_miou) * 100 if baseline_miou > 0 else 0
                sign = '+' if improvement >= 0 else ''
                print(f"  实验{item['experiment']} ({item['mode']}): "
                      f"{sign}{improvement*100:.2f}% ({sign}{improvement_pct:.1f}%)")
    
    print("\n" + "="*80)
    print("✅ 消融实验完成")
    print("="*80)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Pascal-5i批量消融实验（新格式）')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Pascal-5i数据集根目录')
    parser.add_argument('--k_shot', type=int, default=1,
                       help='K-shot设置')
    parser.add_argument('--fold', type=int, default=None,
                       help='评估特定fold (0-3)，默认评估所有folds')
    parser.add_argument('--dino_model', type=str, default='dinov3_vitb16',
                       help='DINO模型')
    parser.add_argument('--sam2_model', type=str, default='large',
                       help='SAM2模型')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--max_query', type=int, default=None,
                       help='每个类别最大query数量（用于快速测试）')
    parser.add_argument('--sorted_support_dir', type=str, default=None,
                       help='预排序support文件基础目录')
    parser.add_argument('--single_mode', type=str, default=None,
                       choices=['rough_only', 'cmrs_predictor', 'memory_only', 'cmrs_memory'],
                       help='只运行单个模式（不进行完整消融实验）')
    
    args = parser.parse_args()
    
    # 如果指定了单个模式
    if args.single_mode:
        modes = [(args.single_mode, f'{args.single_mode} 模式')]
    else:
        modes = None
    
    batch_evaluate(
        args.data_root,
        k_shot=args.k_shot,
        fold=args.fold,
        dino_model=args.dino_model,
        sam2_model=args.sam2_model,
        device=args.device,
        random_seed=args.random_seed,
        max_query=args.max_query,
        sorted_support_dir=args.sorted_support_dir,
        modes=modes
    )


if __name__ == '__main__':
    main()
