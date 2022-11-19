# Libraries
import pytest
import os
import pandas as pd

from structureddata.samplebias.views import samplebias_calculations
from structureddata.misc.preprocessing import preprocess_data


# Fairness Dataset directory
TEST_DATASET_DIR = os.path.join(os.pardir, 'Test DataSets/Fairness')
#print(TEST_DATASET_DIR)
common_dataset_path = os.path.join(TEST_DATASET_DIR,'Structured/Common Data Files')

data_file = os.path.abspath(os.path.join(common_dataset_path,'Set 1/adult_test.csv'))

selected_features = ['workclass','education', 'marital-status', 'occupation', 'relationship']
width = 1920
score_threshold = 0.4

# Sample Bias Unit test case

@pytest.mark.general
@pytest.mark.major
@pytest.mark.parametrize('data_file, selected_features, score_threshold, expected_output', 
                            [
                                (data_file, selected_features, score_threshold, [True, True, True])
                            ]
                        )

def test_samplebias(data_file, selected_features, score_threshold, expected_output):

    data = pd.read_csv(data_file)
    assert isinstance(data, pd.DataFrame) == expected_output[0], 'Datafile is not readable.'

    # Checking pre-process module
    preprocessed_df = preprocess_data(data[selected_features])
    assert isinstance(preprocessed_df, pd.DataFrame) == expected_output[1], 'Object returned by \'preprocess_data\' module is not a pandas dataframe.'
    
    # Checking main module
    samplebias_output, html_fig_str, sample_graph = samplebias_calculations(preprocessed_df,selected_features, width, score_threshold)
    assert isinstance(samplebias_output, list) == expected_output[2], 'Sample Bias output is not a list.'
    
    assert sample_graph in [0,1], 'Seems to be some problem with graph generation module (graph generation indicator).'

    if sample_graph == 1:
        assert isinstance(html_fig_str, str), 'Graph string is not getting generated.'
    else:
        assert isinstance(html_fig_str, None), 'Seems to be some problem with graph generation module (graph html string).'