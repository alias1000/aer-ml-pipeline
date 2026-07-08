"""
Data validation module for AER dataset.

Validates raw dataset for:
- Target variable existence and sanity
- Missing values, duplicates, constant columns
- Categorical variable validity
- Suspicious numerical ranges
"""

import pandas as pd
import json
from pathlib import Path


def validate_target_column(df):
    """Check if AER target column exists and is valid."""
    issues = []
    
    if 'AER' not in df.columns:
        issues.append("Target column 'AER' not found")
        return issues
    
    if df['AER'].isnull().any():
        issues.append(f"AER contains {df['AER'].isnull().sum()} missing values")
    
    if (df['AER'] < 0).any():
        neg_count = (df['AER'] < 0).sum()
        issues.append(f"AER contains {neg_count} negative values")
    
    return issues


def validate_missing_values(df):
    """Check for missing values across all columns."""
    missing = df.isnull().sum()
    issues = []
    
    if missing.sum() > 0:
        missing_cols = missing[missing > 0]
        for col, count in missing_cols.items():
            issues.append(f"Column '{col}' has {count} missing values")
    
    return issues


def validate_duplicates(df):
    """Check for duplicate rows."""
    issues = []
    dup_count = df.duplicated().sum()
    
    if dup_count > 0:
        issues.append(f"Found {dup_count} duplicate rows")
    
    return issues


def validate_categorical_values(df):
    """Validate categorical columns have expected values."""
    issues = []
    
    # Check inlet_angle
    if 'inlet_angle' in df.columns:
        valid_angles = {0, 45, 90}
        invalid = set(df['inlet_angle'].unique()) - valid_angles
        if invalid:
            issues.append(f"inlet_angle contains invalid values: {invalid}")
    
    # Check window
    if 'window' in df.columns:
        valid_windows = {'Open', 'OpenDown', 'OpenUp'}
        invalid = set(df['window'].unique()) - valid_windows
        if invalid:
            issues.append(f"window contains invalid values: {invalid}")
    
    # Check rich_num
    if 'rich_num' in df.columns:
        valid_rich = {'ORI', 'URI', 'ZRI'}
        invalid = set(df['rich_num'].unique()) - valid_rich
        if invalid:
            issues.append(f"rich_num contains invalid values: {invalid}")
    
    return issues


def validate_constant_columns(df):
    """Check for columns with only one unique value."""
    issues = []
    
    for col in df.columns:
        if df[col].nunique() == 1:
            value = df[col].iloc[0]
            issues.append(f"Column '{col}' is constant with value: {value}")
    
    return issues


def validate_numerical_ranges(df):
    """Check for suspicious ranges in numerical columns."""
    issues = []
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numeric_cols:
        if col == 'AER':
            continue
        
        # Check for extreme outliers (beyond typical scientific bounds)
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 10 * iqr  # Generous outlier threshold
        upper_bound = q3 + 10 * iqr
        
        if (df[col] < lower_bound).any() or (df[col] > upper_bound).any():
            issues.append(f"Column '{col}' has potential extreme outliers")
    
    return issues


def generate_validation_report(df):
    """Generate comprehensive validation report."""
    report = {
        'dataset_shape': {'rows': int(df.shape[0]), 'columns': int(df.shape[1])},
        'columns': list(df.columns),
        'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'missing_values': {col: int(count) for col, count in df.isnull().sum().items()},
        'total_missing_cells': int(df.isnull().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum()),
        'constant_columns': [col for col in df.columns if df[col].nunique() == 1],
        'aer_statistics': {
            'min': float(df['AER'].min()),
            'max': float(df['AER'].max()),
            'mean': float(df['AER'].mean()),
            'median': float(df['AER'].median()),
            'std': float(df['AER'].std()),
        },
        'validation_issues': []
    }
    
    # Run all validations
    issues = []
    issues.extend(validate_target_column(df))
    issues.extend(validate_missing_values(df))
    issues.extend(validate_duplicates(df))
    issues.extend(validate_categorical_values(df))
    issues.extend(validate_constant_columns(df))
    issues.extend(validate_numerical_ranges(df))
    
    report['validation_issues'] = issues
    report['is_valid'] = len(issues) == 0
    
    return report


def validate_data(input_path, output_path):
    """Main validation function."""
    print("Loading raw dataset...")
    df = pd.read_csv(input_path)
    
    print("Validating dataset...")
    report = generate_validation_report(df)
    
    # Save report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Validation complete. Report saved to {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"Rows: {report['dataset_shape']['rows']}")
    print(f"Columns: {report['dataset_shape']['columns']}")
    print(f"Missing values: {report['total_missing_cells']}")
    print(f"Duplicates: {report['duplicate_rows']}")
    print(f"Constant columns: {report['constant_columns']}")
    print(f"Issues found: {len(report['validation_issues'])}")
    
    if report['validation_issues']:
        print("\nValidation Issues:")
        for issue in report['validation_issues']:
            print(f"  - {issue}")
    else:
        print("\n[OK] Dataset passed all validation checks!")
    
    print("="*80 + "\n")
    
    return report
