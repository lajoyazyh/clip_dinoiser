from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .types import SliceFindingResult

logger = logging.getLogger(__name__)

class ToolBenchSliceFinder:
    """
    NLP / ToolBench 模态专属切分器：
    不干扰原有视觉模块的软聚类(GMM/SoftKMeans)，独立处理基于 7 维指标的提取逻辑。
    对齐大系统的推荐预测管线中对 `SliceFindingResult` 的标准需求。
    """
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        
    def _preprocess_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        解决连续特征与类别特征在欧氏距离中的失真问题：
        1. 连续特征：Z-Score (StandardScaler)
        2. 离散类别特征：One-Hot Encoding
        """
        df_copy = df.copy().fillna(0)
        
        # 离散/类别特征列表
        discrete_cols = ['quality_outcome', 'difficulty_topology', 'coverage_domain']
        continuous_cols = ['difficulty_length', 'difficulty_instruction_complexity', 'difficulty_ifd', 'coverage_density']
        
        discrete_cols = [c for c in discrete_cols if c in df_copy.columns]
        continuous_cols = [c for c in continuous_cols if c in df_copy.columns]
        
        # 1. 连续特征标准化
        continuous_scaled = self.scaler.fit_transform(df_copy[continuous_cols]) if continuous_cols else np.empty((len(df_copy), 0))
        
        # 2. 离散类别特征 One-Hot 编码 (转换为字符串确保分类)
        df_discrete = df_copy[discrete_cols].astype(str)
        df_dummies = pd.get_dummies(df_discrete)
        
        # 3. 拼接
        if continuous_cols and discrete_cols:
            scaled_features = np.hstack((continuous_scaled, df_dummies.values.astype(float)))
        elif continuous_cols:
            scaled_features = continuous_scaled
        else:
            scaled_features = df_dummies.values.astype(float)
            
        return scaled_features

    def discover_slices(self, sample_ids: List[str], feature_matrix: pd.DataFrame) -> SliceFindingResult:
        """
        执行切分并返回符合当前后端系统标准流的 SliceFindingResult 数据对象。
        """
        if len(sample_ids) != len(feature_matrix):
            raise ValueError("Size of sample_ids must match number of rows in feature_matrix")
            
        logger.info(f"ToolBenchSliceFinder starting extraction... clusters={self.n_clusters}")
        
        # 1. 特征工程预处理
        X = self._preprocess_features(feature_matrix)
        
        # 2. KMeans 硬聚类
        labels = self.kmeans.fit_predict(X)
        centers = self.kmeans.cluster_centers_
        
        # 3. 构建大系统管线需要的软归属概率矩阵 (membership)
        # 对于硬聚类，membership就是One-hot矩阵
        n_samples = len(sample_ids)
        membership = np.zeros((n_samples, self.n_clusters), dtype=np.float32)
        membership[np.arange(n_samples), labels] = 1.0
        
        # 4. 计算每个切片的全局权重 (slice_weights)
        slice_weights = membership.mean(axis=0).astype(np.float32)
        
        result = SliceFindingResult(
            sample_ids=sample_ids,
            membership=membership,
            hard_assignment=labels.astype(np.int64),
            slice_weights=slice_weights,
            centers=centers.astype(np.float32),
            diagnostics={
                "modality": "nlp_toolbench",
                "method": "kmeans_with_onehot_isolation",
                "inertia": self.kmeans.inertia_
            }
        )
        logger.info("ToolBench NLP feature matrix successfully mapped to canonical SliceFindingResult.")
        return result

    def build_projected_features(self, sample_ids: List[str], df: pd.DataFrame) -> Any:
        """
        Build a ProjectedSliceFeatures compatible with the team's downstream Beam Search pipeline.
        Must use dot '.' instead of '_' in dimension prefixes (e.g., 'quality.outcome' instead of 'quality_outcome')
        because prior_graph.py splits on '.' to map feature banks into quality/difficulty/coverage subgroups.
        """
        from .types import ProjectedSliceFeatures

        df_copy = df.copy().fillna(0)
        
        # Mapping our underscored names to dotted names for PriorGraph compatibility
        # We process each column, generate a numpy array slice, and track ranges
        matrix_blocks = []
        block_ranges = {}
        current_col_start = 0
        
        for col in df_copy.columns:
            if "_" in col:
                parts = col.split("_", 1)
                # Enforce valid dimension prefixes
                if parts[0] in ["quality", "difficulty", "coverage"]:
                    block_name = f"{parts[0]}.{parts[1]}"
                else:
                    block_name = col
            else:
                block_name = col
                
            # Assume 1D column (or dummy encoded)
            # For simplicity, if we pass our processed DataFrame before one-hot here:
            col_data = df_copy[[col]].values.astype(np.float32)
            width = col_data.shape[1]
            
            matrix_blocks.append(col_data)
            block_ranges[block_name] = (current_col_start, current_col_start + width)
            current_col_start += width
            
        final_matrix = np.hstack(matrix_blocks) if matrix_blocks else np.empty((len(df_copy), 0))
        
        return ProjectedSliceFeatures(
            matrix=final_matrix,
            sample_ids=sample_ids,
            block_ranges=block_ranges
        )
