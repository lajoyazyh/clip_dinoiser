import os
import json
import glob
import numpy as np
import joblib

def calculate_metrics_for_inference(ans, query):
    valid_data = ans.get('valid_data', False)
    steps = ans.get('intermediate_steps', [])
    has_error = False
    tool_calls = []
    for s in steps:
        if 'tool_name' in s:
            tool_calls.append(s['tool_name'])
            obs = str(s.get('observation', '')).lower()
            if 'error' in obs or s.get('error', False) or '{"error":' in obs:
                has_error = True
                
    res_sc = 1.0 if valid_data and not has_error else (0.5 if valid_data else 0.0)
    clen = min(len(tool_calls)/10.0, 1.0)
    
    import collections
    tcs = list(collections.Counter(tool_calls).values())
    mrep = max(tcs) if tcs else 0
    if mrep <= 1: top = 0.0
    elif mrep == 2: top = 0.5
    else: top = 1.0
        
    words = query.split()
    l_factor = min(len(words)/100.0, 1.0)
    kws = {'if', 'when', 'except', 'sorted', 'filter'}
    c_factor = min(sum(1 for w in words if w.lower() in kws)/3.0, 1.0)
    comp = (l_factor + c_factor)/2.0
    
    def ext(tn): return tn.split("_for_")[-1] if "_for_" in tn else tn
    ut = set(tool_calls)
    ut2 = set(ext(t) for t in ut)
    if len(ut)<=1: dom=0.0
    elif len(ut2)==1: dom=0.3
    elif len(ut2)<=3: dom=0.6
    else: dom=1.0
    
    return [res_sc, clen, top, comp, dom]

def extract_slice_features(inferences_dir):
    inf_files = glob.glob(os.path.join(inferences_dir, "*_CoT@1.json"))
    if not inf_files: return None
    
    metrics_list = []
    for f in inf_files:
        try:
            with open(f, "r") as fp:
                d = json.load(fp)
                ans = d.get('answer_generation', {})
                query = ans.get('query', d.get('query', ''))
                metrics = calculate_metrics_for_inference(ans, query)
                metrics_list.append(metrics)
        except Exception:
            continue
            
    if not metrics_list: return None
    
    # metrics consist of [Result_Score, Chain_Length, Topology_Score, Complexity, Domain_Span]
    mean_metrics = np.mean(metrics_list, axis=0).tolist()
    mean_metrics.append(0.5) # Pad missing local_density
    return np.array(mean_metrics).reshape(1, -1)

class SliceQualityPredictor:
    def __init__(self, model_path='paper_quality_predictor.pkl'):
        # Attempt relative load or absolute
        if not os.path.exists(model_path):
            base_dir = os.path.dirname(__file__)
            model_path = os.path.join(base_dir, model_path)
        self.model = joblib.load(model_path)
        
    def predict_from_inferences(self, inferences_dir):
        X = extract_slice_features(inferences_dir)
        if X is None: return None
        return self.model.predict(X)[0]
        
    def predict_from_features(self, features_array):
        if len(features_array) == 5:
            features_array = list(features_array) + [0.5] # pad density
        return self.model.predict(np.array(features_array).reshape(1, -1))[0]

if __name__ == '__main__':
    print("Testing functionality: Model imports successfully.")
