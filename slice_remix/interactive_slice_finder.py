import os
import json
import glob
import numpy as np
import sys
import random
import subprocess
import time

sys.path.append(os.path.dirname(__file__))
from slice_quality_predictor import calculate_metrics_for_inference, SliceQualityPredictor

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def load_global_pool():
    print(">> 系统初始化: 正在提取全局 Query 特征...")
    root_dir = get_project_root()
    inf_files = glob.glob(os.path.join(root_dir, "data/surrogate_data/*/inferences/*_CoT@1.json"))
    
    pool = []
    base_map = {}
    for bp in glob.glob(os.path.join(root_dir, "code/StableToolBench/solvable_queries/test_instruction/*.json")) + glob.glob(os.path.join(root_dir, "code/StableToolBench/solvable_queries_example/test_instruction/*.json")):
        try:
            with open(bp, 'r') as fp:
                bd = json.load(fp)
                if isinstance(bd, list):
                    for item in bd:
                        if 'query' in item: base_map[item['query']] = item
        except: pass

    for f in inf_files:
        try:
            with open(f, "r") as fp:
                d = json.load(fp)
                ans = d.get('answer_generation', {})
                query = ans.get('query', d.get('query', ''))
                if query not in base_map: continue
                metrics = calculate_metrics_for_inference(ans, query)
                pool.append({
                    "query": query,
                    "file_path": f,
                    "metrics": metrics,
                    "raw_data": base_map[query]
                })
        except:
            pass
    return pool

def recommend_best_feature_ranges(global_pool, predictor):
    best_pred = 0
    best_limits = {}
    
    for _ in range(100):
        limits = {
            "chain_min": random.uniform(0.0, 0.4),
            "chain_max": random.uniform(0.6, 1.0),
            "topo_min": random.uniform(0.0, 0.3),
            "topo_max": random.uniform(0.7, 1.0),
            "comp_min": random.uniform(0.0, 0.5),
            "comp_max": random.uniform(0.5, 1.0),
            "dom_min": random.uniform(0.0, 0.5),
            "dom_max": random.uniform(0.5, 1.0)
        }
        
        filtered = []
        for item in global_pool:
            c, t, comp, dom = item["metrics"][1], item["metrics"][2], item["metrics"][3], item["metrics"][4]
            if (limits["chain_min"] <= c <= limits["chain_max"] and 
                limits["topo_min"] <= t <= limits["topo_max"] and
                limits["comp_min"] <= comp <= limits["comp_max"] and
                limits["dom_min"] <= dom <= limits["dom_max"]):
                filtered.append(item)
                
        if len(filtered) < 10: continue
        
        sub_features = [item["metrics"] for item in filtered]
        sub_mean = np.mean(sub_features, axis=0).tolist() + [0.5]
        pred = predictor.predict_from_features(sub_mean)
        
        if pred > best_pred:
            best_pred = pred
            best_limits = limits
            
    return best_limits, best_pred

def load_recommendations(predictor):
    root_dir = get_project_root()
    manifest_path = os.path.join(root_dir, "surrogate_training_mixtures/mixture_manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    candidates = []
    for item in manifest:
        dist = item.get("feature_distribution", {}).get("mean", {})
        if not dist: continue
        features = [
            dist.get("Result_Score", 0.0), dist.get("Chain_Length", 0.0), 
            dist.get("Topology_Score", 0.0), dist.get("Complexity", 0.0), 
            dist.get("Domain_Span", 0.0), 0.5
        ]
        pred_pr = predictor.predict_from_features(features)
        sample_size = len(item.get("sampled_ids", []))
        candidates.append((item["mixture_id"], pred_pr, features, sample_size))
        
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:3]  

def run_physical_evaluation(mix_id, target_dir, num_samples):
    print(f"\n🚀 [实时任务监控] 正在向物理显卡投递训练/验证任务: {mix_id}")
    print(f"数据量: {num_samples} 独立请求正在启动...")
    try:
        out_dir = os.path.join(get_project_root(), f"data/surrogate_data/{mix_id}/inferences")
        os.makedirs(out_dir, exist_ok=True)
        
        target_count = num_samples
        start_count = len(glob.glob(os.path.join(out_dir, "*_CoT@1.json")))
        print("-------------------------------------------------------")
        
        if start_count >= target_count:
            print(f"⏳ [Live Progress] 发现离线缓存数据已齐全 ({start_count}/{target_count})，跳过重复推断生成。")
        else:
            cmd = ["bash", "run_vllm_task_node.sh", mix_id, target_dir]
            proc = subprocess.Popen(cmd, cwd=get_project_root(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            while proc.poll() is None:
                current_count = len(glob.glob(os.path.join(out_dir, "*_CoT@1.json")))
                display_count = min(current_count, target_count)
                safe_pct = (display_count / target_count * 100) if target_count > 0 else 100.0
                sys.stdout.write(f"\r⏳ [Live Progress] 推理生成中: {display_count}/{target_count} ({safe_pct:.1f}%)   ")
                sys.stdout.flush()
                if current_count >= target_count:
                    break
                time.sleep(2.0)
            proc.wait()
            
        final_files = glob.glob(os.path.join(out_dir, "*_CoT@1.json"))
        passed = 0
        total = len(final_files)
        for f in final_files:
            try:
                with open(f, "r") as fp:
                    d = json.load(fp)
                    ans_gen = d.get('answer_generation', {})
                    c_valid = ans_gen.get('valid_data', False)
                    c_err = False
                    for step in ans_gen.get('intermediate_steps', []):
                        if 'error' in str(step.get('observation','')).lower():
                            c_err = True
                    if c_valid and not c_err:
                        passed += 1
            except:
                pass
                
        real_pr = (passed / total) if total > 0 else 0.0
        
        print(f"\n🎉 [评测完成] 物理训练验证结束！")
        print(f"✅ 测得绝对物理真实通过率: {real_pr:.2%}")
        print("-------------------------------------------------------")
        return real_pr
    except Exception as e:
        print(f"\n❌ [错误] 物理执行异常: {str(e)}")
        return 0.0
def main():
    root_dir = get_project_root()
    predictor = SliceQualityPredictor(os.path.join(os.path.dirname(__file__), 'paper_quality_predictor.pkl'))
    global_pool = load_global_pool()
    
    all_features = [item["metrics"] for item in global_pool]
    baseline_mean = np.mean(all_features, axis=0).tolist() + [0.5]
    baseline_pred = predictor.predict_from_features(baseline_mean)
    
    print("\n=======================================================")
    print("      AI SliceFinder System (Human-in-the-loop)      ")
    print("=======================================================")
    print(f"当前全局数据湖大小: {len(global_pool)} 样本")
    print(f"[Baseline] 大盘随机混合预期平均通过率: {baseline_pred:.2%}")
    print("-------------------------------------------------------\n")
    
    while True:
        recs = load_recommendations(predictor)
        print("\n>> Surrogate Model (预测模型) 为您推荐的静态候选 Slice (Top-3):")
        for i, (mix_id, pr, feats, sample_size) in enumerate(recs):
            print(f"  [{i+1}] {mix_id} | 预期: {pr:.2%} | (Chain={feats[1]:.2f}, Topo={feats[2]:.2f}, Comp={feats[3]:.2f}, Dom={feats[4]:.2f})")
        print("-------------------------------------------------------\n")
        
        print("请选择操作:")
        print("1. 接受以上静态推荐并跑真实评测")
        print("2. 自定义多维参数范围 (系统会先给出参数推荐)")
        print("3. 退出")
        choice = input("请输入选项 (1/2/3): ")
        
        if choice == '3':
            print("系统退出，再见！")
            break
        elif choice == '1':
            sel = int(input("选择推荐编号 (1/2/3): ")) - 1
            rec_mix_id = recs[sel][0]
            rec_pred = recs[sel][1]
            rec_num = recs[sel][3]
            
            mix_dir = f"surrogate_training_mixtures/{rec_mix_id}"
            
            print(f"\n[启动流水线] 开始对推荐集 {rec_mix_id} 进行物理测算 (预测通过率: {rec_pred:.2%}, 数据量: {rec_num})")
            run_physical_evaluation(rec_mix_id, mix_dir, rec_num)
            
            _ = input("\n[操作完成] 按回车键继续下一轮洞察探索...")
            
        elif choice == '2':
            print("\n[系统推演中...] 正在为您搜索能取得极高通过率的潜藏特征边界组合...")
            best_lim, best_pred = recommend_best_feature_ranges(global_pool, predictor)
            print(f"💡 系统智能推荐: 如果您将范围限制在以下区间，预计可达到 {best_pred:.2%} 的超高通过率！")
            if best_lim:
                print(f"  - Chain Length:     [{best_lim['chain_min']:>4.2f}, {best_lim['chain_max']:>4.2f}]")
                print(f"  - Topology Complex: [{best_lim['topo_min']:>4.2f}, {best_lim['topo_max']:>4.2f}]")
                print(f"  - Instruction Comp: [{best_lim['comp_min']:>4.2f}, {best_lim['comp_max']:>4.2f}]")
                print(f"  - Domain Span:      [{best_lim['dom_min']:>4.2f}, {best_lim['dom_max']:>4.2f}]")
            
            print("\n>> 交互式多维特征过滤器 (请按提示输入，直接回车则使用上述推荐值 或 0.0~1.0 满区间)")
            try:
                def get_inp(prompt, default_val):
                    val = input(prompt)
                    return float(val) if val.strip() else default_val
                    
                min_c = get_inp(f"[1/4] Chain Length 最小阈值 (推荐 {best_lim.get('chain_min', 0.0):.2f}): ", best_lim.get('chain_min', 0.0))
                max_c = get_inp(f"      Chain Length 最大阈值 (推荐 {best_lim.get('chain_max', 1.0):.2f}): ", best_lim.get('chain_max', 1.0))
                
                min_t = get_inp(f"[2/4] Topology Complexity 最小阈值 (推荐 {best_lim.get('topo_min', 0.0):.2f}): ", best_lim.get('topo_min', 0.0))
                max_t = get_inp(f"      Topology Complexity 最大阈值 (推荐 {best_lim.get('topo_max', 1.0):.2f}): ", best_lim.get('topo_max', 1.0))

                min_cp = get_inp(f"[3/4] Instruction Complexity 最小阈值 (推荐 {best_lim.get('comp_min', 0.0):.2f}): ", best_lim.get('comp_min', 0.0))
                max_cp = get_inp(f"      Instruction Complexity 最大阈值 (推荐 {best_lim.get('comp_max', 1.0):.2f}): ", best_lim.get('comp_max', 1.0))

                min_d = get_inp(f"[4/4] Domain Span 最小阈值 (推荐 {best_lim.get('dom_min', 0.0):.2f}): ", best_lim.get('dom_min', 0.0))
                max_d = get_inp(f"      Domain Span 最大阈值 (推荐 {best_lim.get('dom_max', 1.0):.2f}): ", best_lim.get('dom_max', 1.0))
                
                filtered = []
                for item in global_pool:
                    c_len, topo, comp, dom = item["metrics"][1], item["metrics"][2], item["metrics"][3], item["metrics"][4]
                    if (min_c <= c_len <= max_c and min_t <= topo <= max_t and
                        min_cp <= comp <= max_cp and min_d <= dom <= max_d):
                        filtered.append(item)
                
                if not filtered:
                    print("!!! 选定的参数范围内没有候选任务，请重新调整。\n")
                    continue
                    
                sub_features = [item["metrics"] for item in filtered]
                sub_mean = np.mean(sub_features, axis=0).tolist() + [0.5]
                sub_pred = predictor.predict_from_features(sub_mean)
                
                print(f"\n[候选子集生成] 搜索到符合条件的 Candidate 样本: {len(filtered)} 个")
                print(f"--> Surrogate 模型实时预测该子集通过率: {sub_pred:.2%} (Baseline: {baseline_pred:.2%})")
                
                if sub_pred > baseline_pred + 0.05:
                    print(f"😊 [用户洞察验证] 绝佳参数！远高于大盘均线！")
                elif sub_pred >= baseline_pred:
                    print(f"😐 [用户洞察验证] 此划分表现平庸，接近大盘平均水平。")
                else:
                    print(f"😟 [用户洞察验证] 此划分预期低于 Baseline，可能选入了高难度数据区。")
                
                sat = input("\n您对此划分是否满意并准备导出评测？(y/n): ")
                if sat.lower() == 'y':
                    custom_id = f"user_custom_slice_{int(time.time())}"
                    out_dir = os.path.join(root_dir, f"surrogate_training_mixtures/{custom_id}")
                    os.makedirs(out_dir, exist_ok=True)
                    
                    out_path = os.path.join(out_dir, f"{custom_id}_test.json")
                    out_data = [item["raw_data"] for item in filtered]
                    with open(out_path, "w") as fp:
                        json.dump(out_data, fp, indent=2)
                        
                    print(f"\n[Success] {len(out_data)} 个样本已物理导出至: {out_path}")
                    print(f"即将调用底层执行脚本进行物理模型训练与评测验证...\n")
                    
                    real_val = run_physical_evaluation(custom_id, f"surrogate_training_mixtures/{custom_id}", len(out_data))
                    
                    print(f"📊 [指标对齐分析]")
                    print(f"您的专家预测期望值: {sub_pred:.2%}")
                    print(f"显卡物理评测真实值: {real_val:.2%}")
                    print(f"MAE 误差: {abs(sub_pred - real_val):.2%}")
                    
                    _ = input("\n[操作完成] 按回车键继续下一轮洞察探索...")
            except Exception as e:
                print(f"输入错误: {e}")

if __name__ == "__main__":
    main()
