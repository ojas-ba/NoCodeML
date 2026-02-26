"""
Transparent Expert System for hyperparameter optimization.
Uses research-backed heuristic rules with full reasoning transparency.
Every decision is explained to the user.
"""
from typing import Dict, Any, Optional, List


class HyperparameterOptimizer:
    """
    Transparent Expert System for hyperparameter optimization.
    Returns optimized parameters along with step-by-step reasoning
    explaining every adjustment made based on dataset characteristics.
    """
    
    def __init__(self):
        """Initialize optimizer with research-backed defaults."""
        
        # Optimal parameters from academic research and Kaggle competitions
        # Tuned for generalization over training accuracy
        self.optimal_params = {
            'classification': {
                'LogisticRegression': {
                    'C': 0.5,
                    'penalty': 'l2',
                    'solver': 'lbfgs',
                    'max_iter': 1000,
                    'class_weight': 'balanced'
                },
                'RandomForestClassifier': {
                    'n_estimators': 150,
                    'max_depth': 12,
                    'min_samples_split': 8,
                    'min_samples_leaf': 4,
                    'max_features': 'sqrt',
                    'class_weight': 'balanced',
                    'bootstrap': True,
                    'random_state': 42
                },
                'XGBClassifier': {
                    'n_estimators': 150,
                    'max_depth': 5,
                    'learning_rate': 0.05,
                    'subsample': 0.7,
                    'colsample_bytree': 0.7,
                    'gamma': 0.3,
                    'min_child_weight': 5,
                    'reg_alpha': 0.3,
                    'reg_lambda': 1.5,
                    'scale_pos_weight': 1.0,
                    'random_state': 42
                },
                'LGBMClassifier': {
                    'n_estimators': 150,
                    'max_depth': 7,
                    'learning_rate': 0.03,
                    'num_leaves': 40,
                    'subsample': 0.7,
                    'colsample_bytree': 0.7,
                    'min_child_samples': 30,
                    'reg_alpha': 0.3,
                    'reg_lambda': 1.5,
                    'class_weight': 'balanced',
                    'random_state': 42,
                    'verbose': -1
                }
            },
            'regression': {
                'LinearRegression': {
                    'fit_intercept': True,
                    'copy_X': True
                },
                'RandomForestRegressor': {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'max_features': 'sqrt',
                    'bootstrap': True,
                    'random_state': 42
                },
                'XGBRegressor': {
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'gamma': 0.1,
                    'min_child_weight': 3,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.0,
                    'random_state': 42
                },
                'LGBMRegressor': {
                    'n_estimators': 200,
                    'max_depth': 8,
                    'learning_rate': 0.05,
                    'num_leaves': 50,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_samples': 20,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.0,
                    'random_state': 42,
                    'verbose': -1
                }
            }
        }
    
    def adjust_for_dataset(
        self,
        params: Dict[str, Any],
        model_type: str,
        n_samples: int,
        n_features: int,
        class_imbalance_ratio: Optional[float] = None
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Adjust hyperparameters based on dataset characteristics.
        Returns both the adjusted params and a list of reasoning objects
        explaining every change made.
        
        Args:
            params: Base hyperparameters
            model_type: Model type
            n_samples: Number of training samples
            n_features: Number of features
            class_imbalance_ratio: Ratio of majority/minority class (for classification)
        
        Returns:
            Tuple of (adjusted_params, applied_rules) where each rule is:
            {"parameter": str, "original_value": Any, "value": Any, "reason": str}
        """
        adjusted = params.copy()
        applied_rules: List[Dict[str, Any]] = []
        
        # --- Small dataset rules (< 1000 samples) ---
        if n_samples < 1000:
            if 'max_depth' in adjusted:
                original = adjusted['max_depth']
                adjusted['max_depth'] = min(original, 6)
                if adjusted['max_depth'] != original:
                    applied_rules.append({
                        "parameter": "max_depth",
                        "original_value": original,
                        "value": adjusted['max_depth'],
                        "reason": f"Small dataset detected ({n_samples:,} samples < 1,000). Capped tree depth to 6 to prevent memorization/overfitting."
                    })
            if 'n_estimators' in adjusted:
                original = adjusted['n_estimators']
                adjusted['n_estimators'] = min(original, 80)
                if adjusted['n_estimators'] != original:
                    applied_rules.append({
                        "parameter": "n_estimators",
                        "original_value": original,
                        "value": adjusted['n_estimators'],
                        "reason": f"Small dataset detected ({n_samples:,} samples < 1,000). Reduced trees to 80 to prevent memorization/overfitting."
                    })
            if 'min_samples_split' in adjusted:
                original = adjusted['min_samples_split']
                adjusted['min_samples_split'] = max(original, 15)
                if adjusted['min_samples_split'] != original:
                    applied_rules.append({
                        "parameter": "min_samples_split",
                        "original_value": original,
                        "value": adjusted['min_samples_split'],
                        "reason": f"Small dataset ({n_samples:,} samples). Raised min split threshold to 15 so each decision node sees enough data."
                    })
            if 'min_samples_leaf' in adjusted:
                original = adjusted['min_samples_leaf']
                adjusted['min_samples_leaf'] = max(original, 8)
                if adjusted['min_samples_leaf'] != original:
                    applied_rules.append({
                        "parameter": "min_samples_leaf",
                        "original_value": original,
                        "value": adjusted['min_samples_leaf'],
                        "reason": f"Small dataset ({n_samples:,} samples). Raised min leaf samples to 8 to ensure each leaf represents a meaningful pattern."
                    })
            if 'C' in adjusted:
                original = adjusted['C']
                adjusted['C'] = min(original, 0.1)
                if adjusted['C'] != original:
                    applied_rules.append({
                        "parameter": "C",
                        "original_value": original,
                        "value": adjusted['C'],
                        "reason": f"Small dataset ({n_samples:,} samples). Reduced regularization strength (C=0.1) to prevent overfitting to noise."
                    })
            if 'learning_rate' in adjusted:
                original = adjusted['learning_rate']
                adjusted['learning_rate'] = max(original * 0.5, 0.01)
                if adjusted['learning_rate'] != original:
                    applied_rules.append({
                        "parameter": "learning_rate",
                        "original_value": original,
                        "value": round(adjusted['learning_rate'], 4),
                        "reason": f"Small dataset ({n_samples:,} samples). Halved learning rate to {adjusted['learning_rate']:.4f} for more cautious learning steps."
                    })
        
        # --- Medium dataset rules (1000-8000 samples) ---
        elif n_samples < 8000:
            if 'max_depth' in adjusted:
                original = adjusted['max_depth']
                adjusted['max_depth'] = min(original, 10)
                if adjusted['max_depth'] != original:
                    applied_rules.append({
                        "parameter": "max_depth",
                        "original_value": original,
                        "value": adjusted['max_depth'],
                        "reason": f"Medium dataset ({n_samples:,} samples). Capped tree depth to 10 for balanced bias-variance tradeoff."
                    })
            if 'n_estimators' in adjusted:
                original = adjusted['n_estimators']
                adjusted['n_estimators'] = min(original, 150)
                if adjusted['n_estimators'] != original:
                    applied_rules.append({
                        "parameter": "n_estimators",
                        "original_value": original,
                        "value": adjusted['n_estimators'],
                        "reason": f"Medium dataset ({n_samples:,} samples). Kept ensemble size at 150 for good generalization without excessive complexity."
                    })
            if 'min_samples_split' in adjusted:
                original = adjusted['min_samples_split']
                adjusted['min_samples_split'] = max(original, 8)
                if adjusted['min_samples_split'] != original:
                    applied_rules.append({
                        "parameter": "min_samples_split",
                        "original_value": original,
                        "value": adjusted['min_samples_split'],
                        "reason": f"Medium dataset ({n_samples:,} samples). Raised min split to 8 for moderate regularization."
                    })
            if 'min_samples_leaf' in adjusted:
                original = adjusted['min_samples_leaf']
                adjusted['min_samples_leaf'] = max(original, 5)
                if adjusted['min_samples_leaf'] != original:
                    applied_rules.append({
                        "parameter": "min_samples_leaf",
                        "original_value": original,
                        "value": adjusted['min_samples_leaf'],
                        "reason": f"Medium dataset ({n_samples:,} samples). Raised min leaf samples to 5 for moderate overfitting prevention."
                    })
            if 'C' in adjusted:
                original = adjusted['C']
                adjusted['C'] = min(original, 0.5)
                if adjusted['C'] != original:
                    applied_rules.append({
                        "parameter": "C",
                        "original_value": original,
                        "value": adjusted['C'],
                        "reason": f"Medium dataset ({n_samples:,} samples). Applied moderate regularization (C=0.5)."
                    })
            if 'reg_alpha' in adjusted:
                original = adjusted['reg_alpha']
                adjusted['reg_alpha'] = max(original, 0.3)
                if adjusted['reg_alpha'] != original:
                    applied_rules.append({
                        "parameter": "reg_alpha",
                        "original_value": original,
                        "value": adjusted['reg_alpha'],
                        "reason": f"Medium dataset ({n_samples:,} samples). Increased L1 penalty to 0.3 for better feature selection and generalization."
                    })
            if 'reg_lambda' in adjusted:
                original = adjusted['reg_lambda']
                adjusted['reg_lambda'] = max(original, 1.5)
                if adjusted['reg_lambda'] != original:
                    applied_rules.append({
                        "parameter": "reg_lambda",
                        "original_value": original,
                        "value": adjusted['reg_lambda'],
                        "reason": f"Medium dataset ({n_samples:,} samples). Increased L2 penalty to 1.5 to smooth predictions."
                    })
        
        # --- High-dimensional data rules (> 50 features) ---
        if n_features > 50:
            if 'colsample_bytree' in adjusted:
                original = adjusted['colsample_bytree']
                adjusted['colsample_bytree'] = 0.6
                if adjusted['colsample_bytree'] != original:
                    applied_rules.append({
                        "parameter": "colsample_bytree",
                        "original_value": original,
                        "value": 0.6,
                        "reason": f"High dimensionality ({n_features} features > 50). Reduced column sampling to 60% to decorrelate trees and prevent overfitting to noisy features."
                    })
            if 'max_features' in adjusted and adjusted['max_features'] == 'sqrt':
                original = adjusted['max_features']
                new_val = min(int(n_features ** 0.5), 20)
                adjusted['max_features'] = new_val
                applied_rules.append({
                    "parameter": "max_features",
                    "original_value": original,
                    "value": new_val,
                    "reason": f"High dimensionality ({n_features} features). Capped max features to {new_val} (√features, max 20) to reduce noise."
                })
        
        # --- Moderate feature set rules (10-30 features) ---
        if 10 < n_features < 30:
            if 'colsample_bytree' in adjusted:
                original = adjusted.get('colsample_bytree', 0.8)
                new_val = min(original, 0.7)
                if new_val != original:
                    adjusted['colsample_bytree'] = new_val
                    applied_rules.append({
                        "parameter": "colsample_bytree",
                        "original_value": original,
                        "value": new_val,
                        "reason": f"Moderate feature set ({n_features} features). Reduced column sampling to 70% to add diversity between trees."
                    })
        
        # --- Class imbalance rules ---
        if class_imbalance_ratio and class_imbalance_ratio > 2.0:
            if 'scale_pos_weight' in adjusted:
                original = adjusted['scale_pos_weight']
                new_val = min(class_imbalance_ratio, 10.0)
                adjusted['scale_pos_weight'] = new_val
                if new_val != original:
                    applied_rules.append({
                        "parameter": "scale_pos_weight",
                        "original_value": original,
                        "value": round(new_val, 2),
                        "reason": f"Class imbalance detected (ratio {class_imbalance_ratio:.1f}:1). Set positive class weight to {new_val:.2f} so the model pays more attention to the minority class."
                    })
            if 'class_weight' in adjusted:
                original = adjusted['class_weight']
                adjusted['class_weight'] = 'balanced'
                if original != 'balanced':
                    applied_rules.append({
                        "parameter": "class_weight",
                        "original_value": original,
                        "value": "balanced",
                        "reason": f"Class imbalance detected (ratio {class_imbalance_ratio:.1f}:1). Set class_weight to 'balanced' to auto-weight classes inversely proportional to their frequency."
                    })
        
        # --- Large dataset rules (> 10000 samples) ---
        if n_samples > 10000:
            if 'learning_rate' in adjusted:
                original = adjusted['learning_rate']
                new_val = min(original * 1.5, 0.3)
                if new_val != original:
                    adjusted['learning_rate'] = new_val
                    applied_rules.append({
                        "parameter": "learning_rate",
                        "original_value": original,
                        "value": round(new_val, 4),
                        "reason": f"Large dataset ({n_samples:,} samples > 10,000). Increased learning rate to {new_val:.4f} — more data means the model can take bigger steps without overfitting."
                    })
        
        # If no rules were applied, note that defaults are already optimal
        if not applied_rules:
            applied_rules.append({
                "parameter": "—",
                "original_value": "—",
                "value": "—",
                "reason": "Dataset characteristics are within normal ranges. Research-backed defaults are already optimal — no adjustments needed."
            })
        
        return adjusted, applied_rules
    
    def generate_tuning_metadata(
        self,
        model_type: str,
        base_score: float,
        n_samples: int,
        n_features: int,
        applied_rules: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate transparent tuning metadata for the UI.
        No simulated trials — just honest expert-system output.
        
        Args:
            model_type: Model type
            base_score: Actual test score achieved
            n_samples: Number of training samples
            n_features: Number of features
            applied_rules: List of reasoning objects from adjust_for_dataset
        
        Returns:
            Tuning metadata with full reasoning transparency
        """
        rules_that_changed = [r for r in applied_rules if r["parameter"] != "—"]
        
        return {
            'enabled': True,
            'method': 'Heuristic Expert Rules',
            'engine': 'NoCodeML Rules Engine v1.0',
            'rules_evaluated': len(applied_rules),
            'rules_applied': len(rules_that_changed),
            'cv_folds': 5,
            'cv_strategy': 'Stratified K-Fold' if 'Classifier' in model_type else 'K-Fold',
            'test_score': float(base_score),
            'applied_rules': applied_rules,
            'dataset_info': {
                'n_samples': n_samples,
                'n_features': n_features,
            }
        }
    
    def optimize(
        self,
        model_type: str,
        task_type: str,
        n_samples: int,
        n_features: int,
        class_imbalance_ratio: Optional[float] = None,
        user_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get optimized hyperparameters for a model.
        
        Args:
            model_type: Model type (e.g., 'LogisticRegression')
            task_type: 'classification' or 'regression'
            n_samples: Number of training samples
            n_features: Number of features
            class_imbalance_ratio: Ratio of majority/minority class
            user_params: User-provided parameters to override
        
        Returns:
            Dict with 'params' and 'metadata'
        """
        # Get base optimal parameters
        base_params = self.optimal_params.get(task_type, {}).get(model_type, {})
        
        if not base_params:
            return {
                'params': user_params or {},
                'metadata': {
                    'enabled': False,
                    'reason': 'Model type not supported for optimization'
                }
            }
        
        # Adjust for dataset characteristics
        optimized_params, applied_rules = self.adjust_for_dataset(
            base_params,
            model_type,
            n_samples,
            n_features,
            class_imbalance_ratio
        )
        
        # Override with user parameters
        if user_params:
            optimized_params.update(user_params)
        
        return {
            'params': optimized_params,
            'applied_rules': applied_rules,
            'metadata': None  # Will be filled after training with actual score
        }
    
    def finalize_metadata(
        self,
        optimized_params: Dict[str, Any],
        test_score: float,
        n_samples: int,
        n_features: int,
        model_type: str,
        applied_rules: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Complete the tuning metadata with actual results.
        
        Args:
            optimized_params: The parameters used
            test_score: Actual test score achieved
            n_samples: Number of training samples
            n_features: Number of features
            model_type: Model type
            applied_rules: List of reasoning objects from optimization
        
        Returns:
            Complete tuning metadata with full transparency
        """
        if applied_rules is None:
            applied_rules = [{
                "parameter": "—",
                "original_value": "—",
                "value": "—",
                "reason": "No rule data available for this run."
            }]
        
        tuning_data = self.generate_tuning_metadata(
            model_type, test_score, n_samples, n_features, applied_rules
        )
        
        tuning_data['best_params'] = optimized_params
        
        return tuning_data


# Singleton instance
hyperparameter_optimizer = HyperparameterOptimizer()
