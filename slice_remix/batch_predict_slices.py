import os
import json
import numpy as np
import joblib

def extract_features_from_metrics(metrics):
    return np.array([
        metrics.get('mean_final_answer_len', 0),
        metrics.get('mean_step_count', 0),
        metrics.get('mean_tool_call_count', 0),
        metrics.get('mean_obs_len', 0),
        metrics.get('valid_data_ratio', 0),
        metrics.get('total', 0)
    ]).reshape(1, -1)

def batch_predict_all_slices(surrogate_data_dir, model_path):
    model = joblib.load(model_path)
    results = []
    for slice_dir in os.listdir(surrogate_data_dir):
        metrics_path = os.path.join(surrogate_data_dir, slice_dir, "inferences", "pass_rate_results", "metrics.json")
        if not os.path.exists(metrics_path):
            continue
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        X = extract_features_from_metrics(metrics)
        pred = model.predict(X)[0]
        gt = metrics.get('physical_pass_rate', None)
        results.append((slice_dir, gt, pred))
    print(f"{'slice':<25} {'真实通过率':<10} {'预测通过率':<10}")
    print('-'*50)
    for s, gt, pred in results:
        print(f"{s:<25} {gt:<10.3f} {pred:<10.3f}")
    diffs = [abs(gt-pred) for _,gt,pred in results if gt is not None]
    print(f"平均绝对误差: {np.mean(diffs):.4f}")

if __name__ == '__main__':
    batch_predict_all_slices('data/surrogate_data', 'clip_dinoiser/slice_remix/slice_quality_predictor.pkl')
