import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional, Tuple
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
import warnings
warnings.filterwarnings('ignore')

class AdvancedMissingDataHandler:
    """
    Industry-standard missing data handler with advanced imputation methods,
    missing data pattern analysis, and comprehensive validation.
    """
    
    def __init__(self):
        self.imputation_history = {}
        self.label_encoders = {}
        self.fitted_imputers = {}
        self.column_types = {}
        
    def analyze_missing_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive analysis of missing data patterns.
        
        Returns:
            Detailed analysis including MCAR, MAR, MNAR assessment
        """
        analysis = {}
        
        # Basic statistics
        missing_count = data.isnull().sum()
        total_rows = len(data)
        missing_percentage = (missing_count / total_rows) * 100
        
        # Missing data pattern matrix
        missing_pattern = data.isnull().astype(int)
        pattern_counts = missing_pattern.value_counts()
        
        # Correlation between missingness indicators
        missing_corr = missing_pattern.corr()
        
        analysis.update({
            'total_rows': total_rows,
            'total_columns': len(data.columns),
            'missing_counts': missing_count.to_dict(),
            'missing_percentages': missing_percentage.to_dict(),
            'columns_with_missing': missing_count[missing_count > 0].index.tolist(),
            'completely_missing_columns': missing_count[missing_count == total_rows].index.tolist(),
            'rows_with_any_missing': data.isnull().any(axis=1).sum(),
            'rows_with_all_missing': data.isnull().all(axis=1).sum(),
            'unique_missing_patterns': len(pattern_counts),
            'missing_pattern_counts': pattern_counts.to_dict(),
            'missingness_correlation': missing_corr.to_dict()
        })
        
        # Recommend missing data mechanism
        analysis['likely_mechanism'] = self._assess_missing_mechanism(data, missing_corr)
        
        return analysis
    
    def _assess_missing_mechanism(self, data: pd.DataFrame, missing_corr: pd.DataFrame) -> str:
        """Assess likely missing data mechanism (MCAR, MAR, MNAR)."""
        # Simple heuristic assessment
        high_corr_pairs = []
        for i in range(len(missing_corr.columns)):
            for j in range(i+1, len(missing_corr.columns)):
                corr_val = abs(missing_corr.iloc[i, j])
                if corr_val > 0.3:  # Threshold for significant correlation
                    high_corr_pairs.append((missing_corr.columns[i], missing_corr.columns[j], corr_val))
        
        if len(high_corr_pairs) > 0:
            return "MAR (Missing at Random) - High correlation between missing patterns"
        elif data.isnull().sum().sum() / (len(data) * len(data.columns)) < 0.05:
            return "MCAR (Missing Completely at Random) - Low missing rate, random pattern"
        else:
            return "MNAR (Missing Not at Random) - Consider domain knowledge"
    
    def handle_missing_advanced(self, 
                               data: pd.DataFrame,
                               strategy: str = 'adaptive',
                               numeric_strategy: str = 'iterative',
                               categorical_strategy: str = 'mode',
                               custom_strategies: Optional[Dict[str, str]] = None,
                               n_neighbors: int = 5,
                               max_iter: int = 10,
                               random_state: int = 42) -> pd.DataFrame:
        """
        Advanced missing data handling with multiple sophisticated strategies.
        
        Args:
            data: Input DataFrame
            strategy: Overall strategy ('adaptive', 'knn', 'iterative', 'rf', 'mice')
            numeric_strategy: Strategy for numeric columns
            categorical_strategy: Strategy for categorical columns
            custom_strategies: Column-specific strategies
            n_neighbors: Number of neighbors for KNN imputation
            max_iter: Maximum iterations for iterative imputation
            random_state: Random state for reproducibility
            
        Returns:
            DataFrame with imputed values
        """
        df = data.copy()
        self.column_types = self._identify_column_types(df)
        
        if strategy == 'adaptive':
            return self._adaptive_imputation(df, numeric_strategy, categorical_strategy, 
                                           custom_strategies, n_neighbors, max_iter, random_state)
        elif strategy == 'knn':
            return self._knn_imputation(df, n_neighbors)
        elif strategy == 'iterative':
            return self._iterative_imputation(df, max_iter, random_state)
        elif strategy == 'mice':
            return self._mice_imputation(df, max_iter, random_state)
        elif strategy == 'rf':
            return self._random_forest_imputation(df, random_state)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _identify_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Identify column types for appropriate imputation strategies."""
        column_types = {}
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                column_types[col] = 'numeric'
            elif df[col].dtype in ['object', 'category']:
                column_types[col] = 'categorical'
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                column_types[col] = 'datetime'
            else:
                column_types[col] = 'other'
        return column_types
    
    def _adaptive_imputation(self, df: pd.DataFrame, numeric_strategy: str, 
                           categorical_strategy: str, custom_strategies: Optional[Dict[str, str]],
                           n_neighbors: int, max_iter: int, random_state: int) -> pd.DataFrame:
        """Adaptive imputation based on column types and missing patterns."""
        
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
                
            # Check for custom strategy
            if custom_strategies and col in custom_strategies:
                strategy = custom_strategies[col]
            else:
                # Use type-based strategy
                if self.column_types[col] == 'numeric':
                    strategy = numeric_strategy
                elif self.column_types[col] == 'categorical':
                    strategy = categorical_strategy
                else:
                    strategy = 'mode'  # Default for other types
            
            # Apply strategy
            if strategy == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mode':
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col].fillna(mode_val[0], inplace=True)
            elif strategy == 'knn':
                df = self._knn_imputation_single_column(df, col, n_neighbors)
            elif strategy == 'iterative':
                df = self._iterative_imputation_single_column(df, col, max_iter, random_state)
            elif strategy == 'interpolate' and self.column_types[col] == 'numeric':
                df[col] = df[col].interpolate()
        
        return df
    
    def _knn_imputation(self, df: pd.DataFrame, n_neighbors: int) -> pd.DataFrame:
        """KNN-based imputation for mixed data types."""
        # Separate numeric and categorical columns
        numeric_cols = [col for col, dtype in self.column_types.items() if dtype == 'numeric']
        categorical_cols = [col for col, dtype in self.column_types.items() if dtype == 'categorical']
        
        # Handle numeric columns with KNN
        if numeric_cols:
            knn_imputer = KNNImputer(n_neighbors=n_neighbors)
            df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])
            self.fitted_imputers['knn_numeric'] = knn_imputer
        
        # Handle categorical columns with mode
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col].fillna(mode_val[0], inplace=True)
        
        return df
    
    def _iterative_imputation(self, df: pd.DataFrame, max_iter: int, random_state: int) -> pd.DataFrame:
        """Iterative imputation (MICE-like) for numeric data."""
        numeric_cols = [col for col, dtype in self.column_types.items() if dtype == 'numeric']
        categorical_cols = [col for col, dtype in self.column_types.items() if dtype == 'categorical']
        
        # Iterative imputation for numeric columns
        if numeric_cols:
            iterative_imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
            df[numeric_cols] = iterative_imputer.fit_transform(df[numeric_cols])
            self.fitted_imputers['iterative_numeric'] = iterative_imputer
        
        # Mode imputation for categorical columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col].fillna(mode_val[0], inplace=True)
        
        return df
    
    def _mice_imputation(self, df: pd.DataFrame, max_iter: int, random_state: int) -> pd.DataFrame:
        """Multiple Imputation by Chained Equations (MICE)."""
        # For this implementation, we'll use IterativeImputer with RandomForest
        numeric_cols = [col for col, dtype in self.column_types.items() if dtype == 'numeric']
        
        if numeric_cols:
            mice_imputer = IterativeImputer(
                estimator=RandomForestRegressor(n_estimators=10, random_state=random_state),
                max_iter=max_iter,
                random_state=random_state
            )
            df[numeric_cols] = mice_imputer.fit_transform(df[numeric_cols])
            self.fitted_imputers['mice'] = mice_imputer
        
        # Handle categorical columns
        categorical_cols = [col for col, dtype in self.column_types.items() if dtype == 'categorical']
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                # Use most frequent value
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col].fillna(mode_val[0], inplace=True)
        
        return df
    
    def _random_forest_imputation(self, df: pd.DataFrame, random_state: int) -> pd.DataFrame:
        """Random Forest-based imputation."""
        df_imputed = df.copy()
        
        # Handle each column with missing values
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
                
            # Separate features and target
            missing_mask = df[col].isnull()
            feature_cols = [c for c in df.columns if c != col]
            
            # Get complete cases for training
            train_mask = ~df[feature_cols].isnull().any(axis=1)
            X_train = df.loc[train_mask, feature_cols]
            y_train = df.loc[train_mask, col]
            
            if len(X_train) == 0 or y_train.isnull().all():
                continue
            
            # Prepare features for prediction
            X_pred = df.loc[missing_mask, feature_cols]
            
            # Handle mixed data types
            X_train_processed = self._preprocess_features(X_train)
            X_pred_processed = self._preprocess_features(X_pred)
            
            # Choose appropriate model
            if self.column_types[col] == 'numeric':
                model = RandomForestRegressor(n_estimators=10, random_state=random_state)
            else:
                model = RandomForestClassifier(n_estimators=10, random_state=random_state)
                
            # Fit and predict
            try:
                model.fit(X_train_processed, y_train)
                predictions = model.predict(X_pred_processed)
                df_imputed.loc[missing_mask, col] = predictions
            except Exception as e:
                # Fallback to simple imputation
                if self.column_types[col] == 'numeric':
                    df_imputed[col].fillna(df[col].mean(), inplace=True)
                else:
                    mode_val = df[col].mode()
                    if not mode_val.empty:
                        df_imputed[col].fillna(mode_val[0], inplace=True)
        
        return df_imputed
    
    def _preprocess_features(self, X: pd.DataFrame) -> np.ndarray:
        """Preprocess features for ML-based imputation."""
        X_processed = X.copy()
        
        # Handle categorical columns
        for col in X_processed.columns:
            if self.column_types.get(col, 'other') == 'categorical':
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    # Fit on non-null values
                    non_null_values = X_processed[col].dropna()
                    if len(non_null_values) > 0:
                        self.label_encoders[col].fit(non_null_values)
                
                # Transform non-null values
                mask = X_processed[col].notnull()
                if mask.sum() > 0:
                    X_processed.loc[mask, col] = self.label_encoders[col].transform(X_processed.loc[mask, col])
        
        # Fill remaining missing values with mean/mode
        for col in X_processed.columns:
            if X_processed[col].isnull().sum() > 0:
                if self.column_types.get(col, 'other') == 'numeric':
                    X_processed[col].fillna(X_processed[col].mean(), inplace=True)
                else:
                    X_processed[col].fillna(X_processed[col].mode()[0] if not X_processed[col].mode().empty else 0, inplace=True)
        
        return X_processed.values
    
    def validate_imputation(self, original_data: pd.DataFrame, imputed_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate imputation results."""
        validation = {}
        
        # Check if all missing values were handled
        remaining_missing = imputed_data.isnull().sum()
        validation['remaining_missing'] = remaining_missing.to_dict()
        validation['imputation_complete'] = remaining_missing.sum() == 0
        
        # Distribution comparison for numeric columns
        distribution_changes = {}
        for col in original_data.columns:
            if self.column_types.get(col, 'other') == 'numeric':
                orig_mean = original_data[col].mean()
                imp_mean = imputed_data[col].mean()
                orig_std = original_data[col].std()
                imp_std = imputed_data[col].std()
                
                distribution_changes[col] = {
                    'mean_change': abs(imp_mean - orig_mean) / orig_mean if orig_mean != 0 else 0,
                    'std_change': abs(imp_std - orig_std) / orig_std if orig_std != 0 else 0
                }
        
        validation['distribution_changes'] = distribution_changes
        
        return validation
    
    def get_imputation_summary(self) -> Dict[str, Any]:
        """Get summary of imputation process."""
        return {
            'fitted_imputers': list(self.fitted_imputers.keys()),
            'column_types': self.column_types,
            'label_encoders': list(self.label_encoders.keys())
        }


# Example usage and comprehensive demonstration
if __name__ == "__main__":
    # Create complex sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate data with different missing patterns
    data = {
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.lognormal(10, 1, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'experience': np.random.normal(8, 5, n_samples),
        'score': np.random.normal(75, 15, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce missing values with different patterns
    # MCAR pattern
    mcar_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    df.loc[mcar_indices, 'age'] = np.nan
    
    # MAR pattern (missing income depends on education)
    low_edu_mask = df['education'] == 'High School'
    mar_indices = df[low_edu_mask].sample(frac=0.3).index
    df.loc[mar_indices, 'income'] = np.nan
    
    # MNAR pattern (high earners don't report income)
    high_income_mask = df['income'] > df['income'].quantile(0.9)
    df.loc[high_income_mask, 'income'] = np.nan
    
    # Random missing in other columns
    df.loc[np.random.choice(n_samples, size=50, replace=False), 'experience'] = np.nan
    df.loc[np.random.choice(n_samples, size=30, replace=False), 'score'] = np.nan
    
    print("Dataset shape:", df.shape)
    print("\nMissing Data Summary:")
    print(df.isnull().sum())
    print("\n" + "="*60 + "\n")
    
    # Initialize advanced handler
    handler = AdvancedMissingDataHandler()
    
    # Comprehensive missing data analysis
    analysis = handler.analyze_missing_patterns(df)
    print("Missing Data Analysis:")
    for key, value in analysis.items():
        if isinstance(value, dict) and len(value) > 5:
            print(f"{key}: {len(value)} items")
        else:
            print(f"{key}: {value}")
    print("\n" + "="*60 + "\n")
    
    # Test different advanced strategies
    strategies = ['adaptive', 'knn', 'iterative', 'mice', 'rf']
    
    for strategy in strategies:
        print(f"Testing {strategy.upper()} imputation...")
        
        try:
            imputed_df = handler.handle_missing_advanced(df, strategy=strategy)
            validation = handler.validate_imputation(df, imputed_df)
            
            print(f"Imputation complete: {validation['imputation_complete']}")
            print(f"Remaining missing values: {sum(validation['remaining_missing'].values())}")
            
            # Show distribution changes for numeric columns
            if validation['distribution_changes']:
                print("Distribution changes:")
                for col, changes in validation['distribution_changes'].items():
                    print(f"  {col}: Mean change: {changes['mean_change']:.3f}, Std change: {changes['std_change']:.3f}")
            
        except Exception as e:
            print(f"Error with {strategy}: {str(e)}")
        
        print("\n" + "-"*40 + "\n")
    
    # Show imputation summary
    summary = handler.get_imputation_summary()
    print("Imputation Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
