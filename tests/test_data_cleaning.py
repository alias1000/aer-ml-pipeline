"""
Unit tests for data cleaning module.

Tests:
- Duplicate removal
- Constant column removal
- Column name cleaning
- AER column preservation
"""

import pytest
import pandas as pd
import numpy as np
from data_cleaning import (
    clean_column_names,
    remove_duplicates,
    remove_constant_columns,
    remove_redundant_columns
)


class TestDataCleaning:
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample dataframe for testing."""
        return pd.DataFrame({
            'AER': [1.0, 2.0, 3.0, 2.0],
            'Temperature': [20, 21, 22, 21],
            'Constant_Col': ['same', 'same', 'same', 'same'],
            'Box_Num': [1, 2, 1, 2],
            'Box_Ver': [1, 1, 1, 1]  # Perfectly collinear with box_num
        })
    
    def test_column_name_cleaning(self, sample_df):
        """Test that column names are standardized."""
        df = clean_column_names(sample_df)
        assert 'aer' in df.columns
        assert 'temperature' in df.columns
        assert 'constant_col' in df.columns
    
    def test_duplicate_removal(self, sample_df):
        """Test that duplicate rows are removed."""
        df = remove_duplicates(sample_df)
        assert len(df) == 3
        assert df.duplicated().sum() == 0
    
    def test_constant_column_removal(self, sample_df):
        """Test that constant columns are removed."""
        df = remove_constant_columns(sample_df)
        assert 'Constant_Col' not in df.columns
        assert 'AER' in df.columns
    
    def test_redundant_columns_removal(self, sample_df):
        """Test that redundant columns are removed."""
        # Add inlet_angle_sine to simulate existing feature
        sample_df['inlet_angle_sine'] = [0.0, 0.5, 1.0, 0.5]
        df = remove_redundant_columns(sample_df)
        assert 'box_ver' not in df.columns or 'Box_Ver' not in df.columns
        assert 'inlet_angle_sine' not in df.columns
    
    def test_aer_preserved(self, sample_df):
        """Test that AER column is always preserved."""
        df = clean_column_names(sample_df)
        df = remove_duplicates(df)
        df = remove_constant_columns(df)
        
        assert 'aer' in df.columns
        assert len(df) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
