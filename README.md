# data-scientist-concept

# Simple usage
handler = AdvancedMissingDataHandler()
analysis = handler.analyze_missing_patterns(df)
imputed_df = handler.handle_missing_advanced(df, strategy='adaptive')

# Custom strategies per column type
imputed_df = handler.handle_missing_advanced(
    df, 
    strategy='adaptive',
    numeric_strategy='iterative',
    categorical_strategy='mode',
    custom_strategies={'income': 'knn', 'age': 'median'}
)
