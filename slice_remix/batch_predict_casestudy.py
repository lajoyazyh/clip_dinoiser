import os
import json
import numpy as np
import sys

# 将当前目录加入 path，保证能 import slice_quality_predictor
sys.path.append(os.path.dirname(__file__))
from slice_quality_predictor import SliceQualityPredictor

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def run_casestudy():
    print("====== 推断级质量预测: Case Study 验证 ======\n")
    predictor_path = os.path.join(os.path.dirname(__file__), 'paper_quality_predictor.pkl')
    predictor = SliceQualityPredictor(predictor_path)
    
    root_dir = get_project_root()
    manifest_path = os.path.join(root_dir, "surrogate_training_mixtures/mixture_manifest.json")
    if not os.path.exists(manifest_path):
        print(f"Manifest not found at {manifest_path}")
        return
        
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    results = []
    
    print(f"{'Mixture ID':<25} | {'Real Pass Rate':<15} | {'Predicted PR':<15} | {'Abs Error'}")
    print("-" * 75)
    
    for item in manifest:
        mix_id = item["mixture_id"]
        dist = item.get("feature_distribution", {})
        
        # 提取论文指出的 6 维特征作为 Ground Truth 输入
        if "mean" in dist:
            m = dist["mean"]
            features = [
                m.get("Result_Score", 0.0), m.get("Chain_Length", 0.0), 
                m.get("Topology_Score", 0.0), m.get("Complexity", 0.0), 
                m.get("Domain_Span", 0.0), m.get("Local_Density", 0.0)
            ]
        else:
            inferences_dir = os.path.join(root_dir, f"data/surrogate_data/{mix_id}/inferences")
            if not os.path.exists(inferences_dir): continue
            import slice_quality_predictor as sqp
            X = sqp.extract_slice_features(inferences_dir)
            if X is None: continue
            features = X.flatten().tolist()
            
        pred = predictor.predict_from_features(features)
        
        # 为了比对，我们还需要实际的 pass rate
        inf_dir = os.path.join(root_dir, f"data/surrogate_data/{mix_id}/inferences")
        import glob
        inf_files = glob.glob(os.path.join(inf_dir, "*_CoT@1.json"))
        if not inf_files: continue
        total = len(inf_files)
        passed = 0
        for f in inf_files:
            try:
                with open(f, "r") as fp:
                    d = json.load(fp)
                    c_valid = d.get('answer_generation', {}).get('valid_data', False)
                    c_err = False
                    for s in d.get('answer_generation', {}).get('intermediate_steps', []):
                         if 'error' in str(s.get('observation','')).lower(): c_err = True
                    if c_valid and not c_err: passed += 1
            except: pass
        
        if total > 0:
            gt = passed / total
            results.append((mix_id, gt, pred, abs(gt - pred), features))
            print(f"{mix_id:<25} | {gt:<15.3%} | {pred:<15.3%} | {abs(gt - pred):.3%}")
            
    if results:
        mae = np.mean([x[3] for x in results])
        print("-" * 75)
        print(f"Overall Mean Absolute Error (MAE): {mae:.4%}")
        
        # 找出预测最准和最不准的两个 slice 进行分析
        results.sort(key=lambda x: x[3])
        best_case = results[0]
        worst_case = results[-1]
        
        print("\n[Case Study 1: 最易预测的集合 (高置信区间)]")
        print(f"Slice: {best_case[0]} (Error: {best_case[3]:.2%})")
        print(f"- 物理真实质量: {best_case[1]:.2%}")
        print(f"- 预测模型质量: {best_case[2]:.2%}")
        print(f"- 主要特征分布: Result_Score={best_case[4][0]:.2f}, Chain_Len={best_case[4][1]:.2f}, Topology={best_case[4][2]:.2f}")
        
        print("\n[Case Study 2: 最难预测的集合 (低置信区间 / 异常特征群)]")
        print(f"Slice: {worst_case[0]} (Error: {worst_case[3]:.2%})")
        print(f"- 物理真实质量: {worst_case[1]:.2%}")
        print(f"- 预测模型质量: {worst_case[2]:.2%}")
        print(f"- 主要特征分布: Result_Score={worst_case[4][0]:.2f}, Chain_Len={worst_case[4][1]:.2f}, Topology={worst_case[4][2]:.2f}")

if __name__ == '__main__':
    run_casestudy()
