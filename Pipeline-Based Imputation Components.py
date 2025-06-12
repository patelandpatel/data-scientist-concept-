import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
import warnings
warnings.filterwarnings('ignore')

class SmartMissingValueImputer(BaseEstimator, TransformerMixin):
    """
    Smart missing value imputer that adapts strategy based on column characteristics.
    Integrates seamlessly with sklearn pipelines.
    """
    
    def __init__(self, 
                 numeric_strategy='auto',
                 categorical_strategy='most_frequent',
                 datetime_strategy='forward_fill',
                 auto_detect_types=True,
                 missing_threshold=0.5,
                 random_state=42):
        """
        Initialize the smart imputer.
        
        Args:
            numeric_strategy: Strategy for numeric columns ('mean', 'median', 'knn', 'iterative', 'auto')
            categorical_strategy: Strategy for categorical columns ('most_frequent', 'constant')
            datetime_strategy: Strategy for datetime columns ('forward_fill', 'backward_fill')
            auto_detect_types: Whether to automatically detect column types
            missing_threshold: Drop columns with missing percentage above this threshold
            random_state: Random state for reproducible results
        """
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.datetime_strategy = datetime_strategy
        self.auto_detect_types = auto_detect_types
        self.missing_threshold = missing_threshold
        self.random_state = random_state
        
        # Will be set during fit
        self.column_types_ = {}
        self.columns_to_drop_ = []
        self.imputers_ = {}
        self.feature_names_in_ = None
        self.n_features_in_ = None
    
    def _detect_column_types(self, X):
        """Detect column types automatically."""
        column_types = {}
        
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                if pd.api.types.is_numeric_dtype(X[col]):
                    column_types[col] = 'numeric'
                elif pd.api.types.is_datetime64_any_dtype(X[col]):
                    column_types[col] = 'datetime'
                else:
                    column_types[col] = 'categorical'
        else:
            # For numpy arrays, assume all numeric
            for i in range(X.shape[1]):
                column_types[i] = 'numeric'
                
        return column_types
    
    def _choose_numeric_strategy(self, X, col):
        """Choose best numeric imputation strategy based on data characteristics."""
        if isinstance(X, pd.DataFrame):
            data = X[col].dropna()
        else:
            col_idx = col if isinstance(col, int) else list(X.columns).index(col)
            data = pd.Series(X[:, col_idx]).dropna()
        
        if len(data) == 0:
            return 'mean'
        
        missing_ratio = 1 - len(data) / len(X)
        skewness = abs(data.skew()) if len(data) > 2 else 0
        
        # Decision logic
        if missing_ratio > 0.3:
            return 'iterative'  # Better for high missing ratios
        elif skewness > 2:
            return 'median'     # Better for skewed data
        else:
            return 'mean'       # Default for normal-ish data
    
    def fit(self, X, y=None):
        """Fit the imputer to the training data."""
        # Store input validation info
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            self.n_features_in_ = X.shape[1]
        else:
            self.n_features_in_ = X.shape[1]
            self.feature_names_in_ = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Detect column types
        if self.auto_detect_types:
            self.column_types_ = self._detect_column_types(X)
        
        # Identify columns to drop (too many missing values)
        if isinstance(X, pd.DataFrame):
            missing_ratios = X.isnull().sum() / len(X)
            self.columns_to_drop_ = missing_ratios[missing_ratios > self.missing_threshold].index.tolist()
        
        # Fit imputers for each column type
        self._fit_imputers(X, y)
        
        return self
    
    def _fit_imputers(self, X, y):
        """Fit appropriate imputers for different column types."""
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=self.feature_names_in_)
        
        # Remove columns with too many missing values
        df = df.drop(columns=self.columns_to_drop_, errors='ignore')
        
        # Group columns by type
        numeric_cols = [col for col, dtype in self.column_types_.items() 
                       if dtype == 'numeric' and col not in self.columns_to_drop_]
        categorical_cols = [col for col, dtype in self.column_types_.items() 
                           if dtype == 'categorical' and col not in self.columns_to_drop_]
        datetime_cols = [col for col, dtype in self.column_types_.items() 
                        if dtype == 'datetime' and col not in self.columns_to_drop_]
        
        # Fit numeric imputers
        if numeric_cols:
            self._fit_numeric_imputers(df[numeric_cols], y)
        
        # Fit categorical imputers
        if categorical_cols:
            self._fit_categorical_imputers(df[categorical_cols])
        
        # Fit datetime imputers
        if datetime_cols:
            self._fit_datetime_imputers(df[datetime_cols])
    
    def _fit_numeric_imputers(self, X_numeric, y):
        """Fit imputers for numeric columns."""
        for col in X_numeric.columns:
            if X_numeric[col].isnull().sum() == 0:
                continue
                
            # Choose strategy
            if self.numeric_strategy == 'auto':
                strategy = self._choose_numeric_strategy(X_numeric, col)
            else:
                strategy = self.numeric_strategy
            
            # Fit appropriate imputer
            if strategy == 'mean':
                imputer = SimpleImputer(strategy='mean')
                imputer.fit(X_numeric[[col]])
            elif strategy == 'median':
                imputer = SimpleImputer(strategy='median')
                imputer.fit(X_numeric[[col]])
            elif strategy == 'knn':
                # Use KNN on all numeric columns
                imputer = KNNImputer(n_neighbors=5)
                imputer.fit(X_numeric)
            elif strategy == 'iterative':
                imputer = IterativeImputer(random_state=self.random_state, max_iter=10)
                imputer.fit(X_numeric)
            
            self.imputers_[f'numeric_{col}'] = {'imputer': imputer, 'strategy': strategy}
    
    def _fit_categorical_imputers(self, X_categorical):
        """Fit imputers for categorical columns."""
        for col in X_categorical.columns:
            if X_categorical[col].isnull().sum() == 0:
                continue
                
            if self.categorical_strategy == 'most_frequent':
                imputer = SimpleImputer(strategy='most_frequent')
            else:
                imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
            
            imputer.fit(X_categorical[[col]])
            self.imputers_[f'categorical_{col}'] = {'imputer': imputer, 'strategy': self.categorical_strategy}
    
    def _fit_datetime_imputers(self, X_datetime):
        """Fit imputers for datetime columns."""
        for col in X_datetime.columns:
            if X_datetime[col].isnull().sum() == 0:
                continue
            
            # Store strategy and parameters for datetime imputation
            self.imputers_[f'datetime_{col}'] = {'strategy': self.datetime_strategy}
    
    def transform(self, X):
        """Transform the data using fitted imputers."""
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=self.feature_names_in_)
        
        # Remove columns with too many missing values
        df = df.drop(columns=self.columns_to_drop_, errors='ignore')
        
        # Apply transformations
        df = self._transform_numeric(df)
        df = self._transform_categorical(df)
        df = self._transform_datetime(df)
        
        return df.values if not isinstance(X, pd.DataFrame) else df
    
    def _transform_numeric(self, df):
        """Transform numeric columns."""
        numeric_cols = [col for col, dtype in self.column_types_.items() 
                       if dtype == 'numeric' and col in df.columns]
        
        for col in numeric_cols:
            if f'numeric_{col}' not in self.imputers_:
                continue
                
            imputer_info = self.imputers_[f'numeric_{col}']
            imputer = imputer_info['imputer']
            strategy = imputer_info['strategy']
            
            if strategy in ['knn', 'iterative']:
                # These imputers work on all numeric columns
                if df[numeric_cols].isnull().sum().sum() > 0:
                    df[numeric_cols] = imputer.transform(df[numeric_cols])
            else:
                # Column-wise imputers
                if df[col].isnull().sum() > 0:
                    df[[col]] = imputer.transform(df[[col]])
        
        return df
    
    def _transform_categorical(self, df):
        """Transform categorical columns."""
        categorical_cols = [col for col, dtype in self.column_types_.items() 
                           if dtype == 'categorical' and col in df.columns]
        
        for col in categorical_cols:
            if f'categorical_{col}' not in self.imputers_:
                continue
                
            imputer = self.imputers_[f'categorical_{col}']['imputer']
            if df[col].isnull().sum() > 0:
                df[[col]] = imputer.transform(df[[col]])
        
        return df
    
    def _transform_datetime(self, df):
        """Transform datetime columns."""
        datetime_cols = [col for col, dtype in self.column_types_.items() 
                        if dtype == 'datetime' and col in df.columns]
        
        for col in datetime_cols:
            if f'datetime_{col}' not in self.imputers_:
                continue
                
            strategy = self.imputers_[f'datetime_{col}']['strategy']
            
            if df[col].isnull().sum() > 0:
                if strategy == 'forward_fill':
                    df[col] = df[col].fillna(method='ffill')
                elif strategy == 'backward_fill':
                    df[col] = df[col].fillna(method='bfill')
        
        return df
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if input_features is None:
            input_features = self.feature_names_in_
        
        # Remove dropped columns
        output_features = [col for col in input_features if col not in self.columns_to_drop_]
        return np.array(output_features)


class ComprehensiveImputationPipeline:
    """
    Comprehensive imputation pipeline that combines multiple strategies
    and provides advanced pipeline construction capabilities.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.pipeline = None
        self.column_info = {}
    
    def create_imputation_pipeline(self, 
                                 X: pd.DataFrame,
                                 y: Optional[pd.Series] = None,
                                 numeric_strategy='auto',
                                 categorical_strategy='most_frequent',
                                 include_feature_engineering=False) -> Pipeline:
        """
        Create a comprehensive imputation pipeline.
        
        Args:
            X: Training data
            y: Target variable (optional)
            numeric_strategy: Strategy for numeric columns
            categorical_strategy: Strategy for categorical columns
            include_feature_engineering: Whether to include feature engineering steps
            
        Returns:
            sklearn Pipeline object
        """
        # Analyze data
        self._analyze_data(X)
        
        # Build pipeline steps
        steps = []
        
        # Step 1: Missing value analysis and reporting
        steps.append(('missing_analyzer', MissingValueAnalyzer()))
        
        # Step 2: Smart imputation
        steps.append(('imputer', SmartMissingValueImputer(
            numeric_strategy=numeric_strategy,
            categorical_strategy=categorical_strategy,
            random_state=self.random_state
        )))
        
        # Step 3: Post-imputation validation
        steps.append(('validator', ImputationValidator()))
        
        # Optional: Feature engineering after imputation
        if include_feature_engineering:
            steps.append(('feature_engineer', PostImputationFeatureEngineer()))
        
        self.pipeline = Pipeline(steps)
        return self.pipeline
    
    def _analyze_data(self, X):
        """Analyze data characteristics for pipeline optimization."""
        self.column_info = {
            'numeric_columns': X.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': X.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime_columns': X.select_dtypes(include=['datetime']).columns.tolist(),
            'missing_percentages': (X.isnull().sum() / len(X) * 100).to_dict(),
            'total_missing': X.isnull().sum().sum(),
            'rows_with_missing': X.isnull().any(axis=1).sum()
        }
    
    def fit_transform(self, X, y=None):
        """Fit the pipeline and transform data."""
        if self.pipeline is None:
            self.create_imputation_pipeline(X, y)
        
        return self.pipeline.fit_transform(X, y)
    
    def transform(self, X):
        """Transform data using fitted pipeline."""
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit_transform first.")
        return self.pipeline.transform(X)
    
    def get_pipeline_info(self):
        """Get information about the fitted pipeline."""
        info = {
            'pipeline_steps': [step[0] for step in self.pipeline.steps] if self.pipeline else [],
            'column_info': self.column_info
        }
        return info


class MissingValueAnalyzer(BaseEstimator, TransformerMixin):
    """Analyzer to capture missing value patterns before imputation."""
    
    def __init__(self):
        self.missing_patterns_ = {}
        self.analysis_results_ = {}
    
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            df = X
        else:
            df = pd.DataFrame(X)
        
        # Capture missing patterns
        self.missing_patterns_ = {
            'missing_counts': df.isnull().sum().to_dict(),
            'missing_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
            'pattern_matrix': df.isnull().astype(int)
        }
        
        return self
    
    def transform(self, X):
        # Pass through unchanged - this is just for analysis
        return X
    
    def get_analysis(self):
        return self.missing_patterns_


class ImputationValidator(BaseEstimator, TransformerMixin):
    """Validator to check imputation quality."""
    
    def __init__(self):
        self.validation_results_ = {}
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Validate that imputation worked
        if isinstance(X, pd.DataFrame):
            df = X
        else:
            df = pd.DataFrame(X)
        
        remaining_missing = df.isnull().sum().sum()
        
        self.validation_results_ = {
            'imputation_successful': remaining_missing == 0,
            'remaining_missing_values': remaining_missing,
            'final_shape': df.shape
        }
        
        if remaining_missing > 0:
            print(f"Warning: {remaining_missing} missing values remain after imputation")
        
        return X


class PostImputationFeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature engineering steps that can be applied after imputation."""
    
    def __init__(self):
        self.engineered_features_ = []
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X)
        
        # Example feature engineering after imputation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Create missing value indicators for originally missing columns
        for col in numeric_cols:
            # This is just an example - in practice, you'd track which values were originally missing
            missing_indicator_col = f'{col}_was_missing'
            if missing_indicator_col not in df.columns:
                # Placeholder - in real implementation, you'd have this info from the analyzer
                df[missing_indicator_col] = 0
                self.engineered_features_.append(missing_indicator_col)
        
        return df.values if not isinstance(X, pd.DataFrame) else df


# Example usage and demonstration
if __name__ == "__main__":
    # Create comprehensive test dataset
    np.random.seed(42)
    n_samples = 1000
    
    # Generate data with realistic missing patterns
    data = {
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.lognormal(10, 1, n_samples),
        'score': np.random.normal(75, 15, n_samples),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'education': np.random.choice(['HS', 'College', 'Graduate'], n_samples),
        'date_col': pd.date_range('2020-01-01', periods=n_samples, freq='D')
    }
    
    df = pd.DataFrame(data)
    
    # Introduce various missing patterns
    # MCAR - Missing Completely at Random
    np.random.seed(42)
    mcar_indices = np.random.choice(n_samples, size=100, replace=False)
    df.loc[mcar_indices, 'age'] = np.nan
    
    # MAR - Missing at Random (correlated with other variables)
    high_income_mask = df['income'] > df['income'].quantile(0.8)
    df.loc[high_income_mask, 'score'] = np.nan
    
    # Categorical missing
    cat_missing_idx = np.random.choice(n_samples, size=80, replace=False)
    df.loc[cat_missing_idx, 'category'] = np.nan
    
    # Date missing
    date_missing_idx = np.random.choice(n_samples, size=50, replace=False)
    df.loc[date_missing_idx, 'date_col'] = np.nan
    
    print("Dataset with Missing Values:")
    print(f"Shape: {df.shape}")
    print(f"Missing values per column:")
    print(df.isnull().sum())
    print(f"Missing value percentages:")
    print((df.isnull().sum() / len(df) * 100).round(2))
    print("\n" + "="*60 + "\n")
    
    # Split data
    train_size = int(0.8 * len(df))
    X_train, X_test = df.iloc[:train_size], df.iloc[train_size:]
    
    print("Testing Different Pipeline Approaches:")
    print("\n" + "="*60 + "\n")
    
    # Test 1: Simple Smart Imputer
    print("1. Simple Smart Imputer Pipeline:")
    
    smart_imputer = SmartMissingValueImputer(
        numeric_strategy='auto',
        categorical_strategy='most_frequent',
        random_state=42
    )
    
    # Create simple pipeline
    simple_pipeline = Pipeline([
        ('imputer', smart_imputer),
        ('scaler', StandardScaler())
    ])
    
    X_train_simple = simple_pipeline.fit_transform(X_train)
    X_test_simple = simple_pipeline.transform(X_test)
    
    print(f"  Training shape: {X_train.shape} → {X_train_simple.shape}")
    print(f"  Test shape: {X_test.shape} → {X_test_simple.shape}")
    print(f"  Missing values remaining: {pd.DataFrame(X_train_simple).isnull().sum().sum()}")
    
    # Test 2: Comprehensive Pipeline
    print("\n2. Comprehensive Imputation Pipeline:")
    
    comprehensive_pipeline = ComprehensiveImputationPipeline(random_state=42)
    
    X_train_comprehensive = comprehensive_pipeline.fit_transform(X_train)
    X_test_comprehensive = comprehensive_pipeline.transform(X_test)
    
    print(f"  Training shape: {X_train.shape} → {X_train_comprehensive.shape}")
    print(f"  Test shape: {X_test.shape} → {X_test_comprehensive.shape}")
    
    # Get pipeline info
    pipeline_info = comprehensive_pipeline.get_pipeline_info()
    print(f"  Pipeline steps: {pipeline_info['pipeline_steps']}")
    print(f"  Numeric columns: {len(pipeline_info['column_info']['numeric_columns'])}")
    print(f"  Categorical columns: {len(pipeline_info['column_info']['categorical_columns'])}")
    
    # Test 3: Custom Column-Specific Pipeline
    print("\n3. Custom Column-Specific Pipeline:")
    
    # Define different strategies for different columns
    numeric_cols = ['age', 'income', 'score']
    categorical_cols = ['category', 'education']
    
    # Create column-specific transformers
    numeric_transformer = Pipeline([
        ('imputer', IterativeImputer(random_state=42, max_iter=10)),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    
    # Combine using ColumnTransformer
    column_specific_pipeline = ColumnTransformer([
        ('numeric', numeric_transformer, numeric_cols),
        ('categorical', categorical_transformer, categorical_cols)
    ], remainder='drop')  # Drop datetime column for this example
    
    X_train_custom = column_specific_pipeline.fit_transform(X_train)
    X_test_custom = column_specific_pipeline.transform(X_test)
    
    print(f"  Training shape: {X_train.shape} → {X_train_custom.shape}")
    print(f"  Test shape: {X_test.shape} → {X_test_custom.shape}")
    print(f"  Features processed: {len(numeric_cols + categorical_cols)}")
    
    print("\n" + "="*60 + "\n")
    print("Pipeline Comparison Summary:")
    print(f"Simple Pipeline:        {X_train_simple.shape[1]} features")
    print(f"Comprehensive Pipeline: {X_train_comprehensive.shape[1]} features") 
    print(f"Custom Pipeline:        {X_train_custom.shape[1]} features")
    
    print("\n" + "="*60 + "\n")
    print("Key Advantages of Pipeline-Based Imputation:")
    print("✓ Ensures consistent train/test processing")
    print("✓ Prevents data leakage by fitting only on training data")
    print("✓ Integrates seamlessly with sklearn workflows")
    print("✓ Enables complex multi-step preprocessing")
    print("✓ Provides reproducible results")
    print("✓ Supports cross-validation and hyperparameter tuning")
    print("✓ Easy to deploy in production environments")
