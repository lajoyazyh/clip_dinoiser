import os
import json
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

def extract_features_from_metrics(metrics):
    return np.array([
        metrics.get('mean_final_answer_len', 0),
        metrics.get('mean_step_count', 0),
        metrics.get('mean_tool_call_count', 0),
        metrics.get('mean_obs_len', 0),
        metrics.get('valid_data_ratio', 0),
        metrics.get('total', 0)
    ]).reshape(1, -1)

def train_and_save_model(surrogate_data_dir, model_path='slice_quality_predictor.pkl'):
    features = []
    labels = []
    for slice_dir in os.listdir(surrogate_data_dir):
        metrics_path = os.path.join(surrogate_data_dir, slice_dir, "inferences", "pass_rate_results", "metrics.json")
        if not os.path.exists(metrics_path):
            continue
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        feat = extract_features_from_metrics(metrics).flatten()
        label = metrics.get('physical_pass_rate', None)
        if label is None:
            continue
        features.append(feat)
        labels.append(label)
    X = np.array(features)
    y = np.array(labels)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, model_path)
    print(f"模型已保存到 {model_path}")

def predict_slice_quality(metrics_json_path, model_path='slice_quality_predictor.pkl'):
    import joblib
    with open(metrics_json_path, 'r') as f:
        metrics = json.load(f)
    X = extract_features_from_metrics(metrics)
    model = joblib.load(model_path)
    pred = model.predict(X)[0]
    print(f"预测物理通过率: {pred:.4f}")
    return pred

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], required=True)
    parser.add_argument('--data_dir', type=str, default='data/surrogate_data')
    parser.add_argument('--model_path', type=str, default='slice_quality_predictor.pkl')
    parser.add_argument('--metrics_json', type=str, help='单个slice的metrics.json路径')
    args = parser.parse_args()
    if args.mode == 'train':
        train_and_save_model(args.data_dir, args.model_path)
    elif args.mode == 'predict':
        if not args.metrics_json:
            print('请指定 --metrics_json')
        else:
            predict_slice_quality(args.metrics_json, args.model_path)
