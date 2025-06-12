import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any

class SimpleMissingDataHandler:
    """
    Simple version for handling missing values in datasets.
    Covers basic imputation strategies for different data types.
    """
    
    def __init__(self):
        self.strategies = {
            'drop': self._drop_missing,
            'mean': self._mean_imputation,
            'median': self._median_imputation,
            'mode': self._mode_imputation,
            'forward_fill': self._forward_fill,
            'backward_fill': self._backward_fill,
            'constant': self._constant_fill,
            'interpolate': self._interpolate
        }
    
    def handle_missing(self, 
                      data: pd.DataFrame, 
                      strategy: str = 'mean',
                      columns: Union[str, List[str], None] = None,
                      fill_value: Any = 0) -> pd.DataFrame:
        """
        Handle missing values using specified strategy.
        
        Args:
            data: Input DataFrame
            strategy: Strategy to use ('drop', 'mean', 'median', 'mode', 
                     'forward_fill', 'backward_fill', 'constant', 'interpolate')
            columns: Specific columns to process (None for all)
            fill_value: Value to use for constant fill strategy
            
        Returns:
            DataFrame with missing values handled
        """
        df = data.copy()
        
        if columns is None:
            columns = df.columns.tolist()
        elif isinstance(columns, str):
            columns = [columns]
            
        if strategy not in self.strategies:
            raise ValueError(f"Strategy must be one of: {list(self.strategies.keys())}")
            
        return self.strategies[strategy](df, columns, fill_value)
    
    def _drop_missing(self, df: pd.DataFrame, columns: List[str], fill_value: Any) -> pd.DataFrame:
        """Drop rows with missing values in specified columns."""
        return df.dropna(subset=columns)
    
    def _mean_imputation(self, df: pd.DataFrame, columns: List[str], fill_value: Any) -> pd.DataFrame:
        """Fill missing values with column mean (numeric columns only)."""
        for col in columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].mean(), inplace=True)
        return df
    
    def _median_imputation(self, df: pd.DataFrame, columns: List[str], fill_value: Any) -> pd.DataFrame:
        """Fill missing values with column median (numeric columns only)."""
        for col in columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
        return df
    
    def _mode_imputation(self, df: pd.DataFrame, columns: List[str], fill_value: Any) -> pd.DataFrame:
        """Fill missing values with column mode (most frequent value)."""
        for col in columns:
            mode_value = df[col].mode()
            if not mode_value.empty:
                df[col].fillna(mode_value[0], inplace=True)
        return df
    
    def _forward_fill(self, df: pd.DataFrame, columns: List[str], fill_value: Any) -> pd.DataFrame:
        """Forward fill missing values."""
        df[columns] = df[columns].fillna(method='ffill')
        return df
    
    def _backward_fill(self, df: pd.DataFrame, columns: List[str], fill_value: Any) -> pd.DataFrame:
        """Backward fill missing values."""
        df[columns] = df[columns].fillna(method='bfill')
        return df
    
    def _constant_fill(self, df: pd.DataFrame, columns: List[str], fill_value: Any) -> pd.DataFrame:
        """Fill missing values with constant value."""
        for col in columns:
            df[col].fillna(fill_value, inplace=True)
        return df
    
    def _interpolate(self, df: pd.DataFrame, columns: List[str], fill_value: Any) -> pd.DataFrame:
        """Interpolate missing values (numeric columns only)."""
        for col in columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].interpolate()
        return df
    
    def analyze_missing(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze missing data patterns in the dataset.
        
        Returns:
            Dictionary with missing data statistics
        """
        analysis = {}
        
        # Basic statistics
        missing_count = data.isnull().sum()
        total_rows = len(data)
        missing_percentage = (missing_count / total_rows) * 100
        
        analysis['missing_counts'] = missing_count.to_dict()
        analysis['missing_percentages'] = missing_percentage.to_dict()
        analysis['total_rows'] = total_rows
        analysis['columns_with_missing'] = missing_count[missing_count > 0].index.tolist()
        
        return analysis


# Example usage and demonstration
if __name__ == "__main__":
    # Create sample data with missing values
    np.random.seed(42)
    sample_data = {
        'numeric_col1': [1, 2, np.nan, 4, 5, np.nan, 7, 8],
        'numeric_col2': [10, np.nan, 30, 40, np.nan, 60, 70, 80],
        'categorical_col': ['A', 'B', np.nan, 'A', 'C', 'B', np.nan, 'C'],
        'date_col': pd.date_range('2023-01-01', periods=8, freq='D'),
        'text_col': ['hello', np.nan, 'world', 'test', np.nan, 'data', 'science', np.nan]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original Data:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Initialize handler
    handler = SimpleMissingDataHandler()
    
    # Analyze missing data
    analysis = handler.analyze_missing(df)
    print("Missing Data Analysis:")
    for key, value in analysis.items():
        print(f"{key}: {value}")
    print("\n" + "="*50 + "\n")
    
    # Demonstrate different strategies
    strategies = ['mean', 'median', 'mode', 'constant', 'drop']
    
    for strategy in strategies:
        print(f"Strategy: {strategy}")
        if strategy == 'constant':
            result = handler.handle_missing(df, strategy=strategy, fill_value='MISSING')
        else:
            result = handler.handle_missing(df, strategy=strategy)
        print(result)
        print("\n" + "-"*30 + "\n")
