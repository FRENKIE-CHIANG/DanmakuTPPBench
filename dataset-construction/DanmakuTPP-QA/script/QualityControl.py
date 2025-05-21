import json
import os
from collections import Counter
from typing import List, Dict, Any, Union
import numpy as np

class QualityControlAgent:
    """
    质量控制代理，用于对多个模型的标注结果进行审核和合并
    """
    def __init__(self, annotation_files: List[str]):
        """
        初始化质量控制代理
        
        Args:
            annotation_files: 包含标注结果的JSON文件路径列表
        """
        self.annotation_files = annotation_files
        self.annotations = self._load_annotations()
        
    def _load_annotations(self) -> List[Dict[str, Any]]:
        """加载所有标注文件"""
        annotations = []
        for file_path in self.annotation_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                annotations.append({"file": file_path, "data": data})
                print(f"成功加载标注文件: {file_path}")
            except Exception as e:
                print(f"加载文件 {file_path} 时出错: {e}")
        return annotations
    
    def majority_voting(self, key_path: Union[str, List[str]]) -> Any:
        """
        使用多数投票策略选择最佳标注
        
        Args:
            key_path: 要在标注数据中访问的键路径，可以是单个键或键的列表
        """
        values = []
        
        # 将单个键转换为列表以便统一处理
        if isinstance(key_path, str):
            key_path = [key_path]
            
        for annotation in self.annotations:
            # 逐层获取嵌套的键值
            value = annotation["data"]
            try:
                for key in key_path:
                    value = value[key]
                # 对于复杂对象（如字典、列表），转换为字符串进行比较
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, sort_keys=True)
                values.append(value)
            except (KeyError, TypeError):
                # 如果键不存在，跳过
                continue
                
        # 如果没有有效值，返回None
        if not values:
            return None
            
        # 使用Counter找出出现次数最多的值
        counter = Counter(values)
        most_common = counter.most_common(1)[0][0]
        
        # 如果是JSON字符串，转换回原始对象
        if isinstance(most_common, str) and (most_common.startswith('{') or most_common.startswith('[')):
            try:
                return json.loads(most_common)
            except:
                return most_common
        
        return most_common
    
    def gap_filling(self, structure_template: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用模板结构，填补缺失的标注
        
        Args:
            structure_template: 定义最终结果结构的模板
        """
        result = {}
        
        def fill_structure(template, target, path=""):
            """递归填充结构"""
            for key, expected_value in template.items():
                current_path = f"{path}.{key}" if path else key
                current_path_list = current_path.split(".")
                
                if isinstance(expected_value, dict):
                    # 如果期望是字典，递归处理
                    target[key] = {}
                    fill_structure(expected_value, target[key], current_path)
                else:
                    # 使用majority_voting策略获取值
                    value = self.majority_voting(current_path_list)
                    
                    # 如果没有找到值，但有期望的类型，则创建空的默认值
                    if value is None and expected_value is not None:
                        if expected_value == list:
                            value = []
                        elif expected_value == dict:
                            value = {}
                        elif expected_value == str:
                            value = ""
                        elif expected_value == int:
                            value = 0
                        elif expected_value == float:
                            value = 0.0
                        elif expected_value == bool:
                            value = False
                    
                    target[key] = value
        
        fill_structure(structure_template, result)
        return result
    
    def confidence_weighted_merge(self, key_path: Union[str, List[str]], confidences: Dict[str, float]) -> Any:
        """
        基于置信度加权合并结果
        
        Args:
            key_path: 要在标注数据中访问的键路径
            confidences: 每个模型的置信度得分
        """
        # 将单个键转换为列表以便统一处理
        if isinstance(key_path, str):
            key_path = [key_path]
            
        # 按值存储加权计数
        weighted_counts = {}
        
        for annotation in self.annotations:
            file_name = os.path.basename(annotation["file"])
            confidence = confidences.get(file_name, 1.0)  # 默认置信度为1
            
            # 获取值
            value = annotation["data"]
            try:
                for key in key_path:
                    value = value[key]
                
                # 处理复杂对象
                if isinstance(value, (dict, list)):
                    value_key = json.dumps(value, sort_keys=True)
                else:
                    value_key = value
                    
                weighted_counts[value_key] = weighted_counts.get(value_key, 0) + confidence
            except (KeyError, TypeError):
                continue
        
        # 如果没有有效值，返回None
        if not weighted_counts:
            return None
            
        # 选择加权分数最高的值
        best_value = max(weighted_counts.items(), key=lambda x: x[1])[0]
        
        # 如果是JSON字符串，转换回原始对象
        if isinstance(best_value, str) and (best_value.startswith('{') or best_value.startswith('[')):
            try:
                return json.loads(best_value)
            except:
                return best_value
        
        return best_value
    
    def agreement_score(self, key_path: Union[str, List[str]]) -> float:
        """
        计算不同模型在特定字段上的一致性得分
        
        Args:
            key_path: 要在标注数据中访问的键路径
        """
        values = []
        
        # 将单个键转换为列表以便统一处理
        if isinstance(key_path, str):
            key_path = [key_path]
            
        for annotation in self.annotations:
            # 逐层获取嵌套的键值
            value = annotation["data"]
            try:
                for key in key_path:
                    value = value[key]
                # 对于复杂对象，转换为字符串
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, sort_keys=True)
                values.append(value)
            except (KeyError, TypeError):
                continue
                
        if not values or len(values) == 1:
            return 0.0
            
        # 计算一致性得分 (出现最多的值的次数 / 总标注数)
        counter = Counter(values)
        most_common_count = counter.most_common(1)[0][1]
        return most_common_count / len(values)
    
    def process_annotations(self, strategy: str = "majority_voting", **kwargs) -> Dict[str, Any]:
        """
        处理标注，应用指定的策略
        
        Args:
            strategy: 使用的策略名称
            **kwargs: 策略可能需要的额外参数
        """
        if strategy == "majority_voting":
            # 处理顶层的所有键
            if not self.annotations or not self.annotations[0]["data"]:
                return {}
            
            sample_data = self.annotations[0]["data"]
            result = {}
            
            # 处理顶层键
            for key in sample_data:
                result[key] = self.majority_voting(key)
                
            return result
            
        elif strategy == "gap_filling":
            if "structure_template" not in kwargs:
                raise ValueError("Gap filling strategy requires a structure_template")
            
            return self.gap_filling(kwargs["structure_template"])
            
        elif strategy == "confidence_weighted":
            if "confidences" not in kwargs:
                raise ValueError("Confidence weighted strategy requires confidences map")
                
            # 处理顶层的所有键
            if not self.annotations or not self.annotations[0]["data"]:
                return {}
            
            sample_data = self.annotations[0]["data"]
            result = {}
            
            # 处理顶层键
            for key in sample_data:
                result[key] = self.confidence_weighted_merge(key, kwargs["confidences"])
                
            return result
            
        elif strategy == "hybrid":
            if "field_strategies" not in kwargs:
                raise ValueError("Hybrid strategy requires field_strategies mapping")
                
            return self.hybrid_strategy(**kwargs)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
    def hybrid_strategy(self, field_strategies: Dict[str, Dict], **kwargs) -> Dict[str, Any]:
        """
        结合多种策略，为不同字段应用不同的处理方法
        
        Args:
            field_strategies: 字段与策略的映射，如 {"field": {"strategy": "strategy_name", "params": {...}}}
        """
        result = {}
        
        for field, strategy_info in field_strategies.items():
            strategy_name = strategy_info["strategy"]
            strategy_params = strategy_info.get("params", {})
            
            # 根据字段的指定策略处理
            if strategy_name == "majority_voting":
                result[field] = self.majority_voting(field)
            elif strategy_name == "confidence_weighted":
                if "confidences" not in strategy_params:
                    raise ValueError(f"Field {field}: Confidence weighted strategy requires confidences")
                result[field] = self.confidence_weighted_merge(field, strategy_params["confidences"])
            elif strategy_name == "gap_filling":
                if "structure_template" not in strategy_params:
                    raise ValueError(f"Field {field}: Gap filling strategy requires structure_template")
                    
                temp_template = {field: strategy_params["structure_template"]}
                filled_result = self.gap_filling(temp_template)
                result[field] = filled_result.get(field)
            else:
                raise ValueError(f"Field {field}: Unknown strategy {strategy_name}")
                
        return result
