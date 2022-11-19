import pytest
import os
import pandas as pd
from structureddata.misc.preprocessing import preprocess_data
from structureddata.assessment.fairness_analysis import generate_dissimilarity_report, generateFairnessReport

# Dataset directory
TEST_DATASET_DIR = "../Test DataSets/Fairness/Structured/Common Data Files/"
TEST_PREDICTIONS_DIR = "../Test DataSets/Fairness/Structured/LabelBias_PostPred/"

correct_req_path1 = os.path.join(TEST_DATASET_DIR,'Set 1/')
correct_req_path2 = os.path.join(TEST_PREDICTIONS_DIR,'Set 1/')
correct_req_path3 = os.path.join(TEST_DATASET_DIR,'Set 2/')
correct_req_path4 = os.path.join(TEST_PREDICTIONS_DIR,'Set 2/')

@pytest.mark.parametrize("datafile, predfile, selected_features, selected_target, target_category, selected_acc_metrics, selected_bias_metrics, bias_threshold, inputs, expected_output", 
                                                                        [(correct_req_path1+'adult_test.csv', correct_req_path2+'predictions.csv', ['race', 'sex'], 'class', '>50K', 'precision_recall', 'distance', 20, ['sex_ Male', 'race_ Black'], ['Biased', 0.235]),
                                                                        (correct_req_path3+'compas.csv', correct_req_path4+'predictions_compas.csv', ['race', 'sex'], 'score_text', 'High', 'tpr_fpr', 'pvalue', 5, ['race_African-American', 'sex_Female'], ['Fair', 0.191])
                                                                        ])

def test_generate_dissimilarity_report_postprediction(datafile, predfile, selected_features, selected_target, target_category, selected_acc_metrics, 
                                                        selected_bias_metrics, bias_threshold, inputs, expected_output):
    '''
    Purpose: This unit test checks generate_dissimilarity_report_postprediction module using 2 different test cases.
    Arguments: Common dataset file, prediction file, selected sensitive features, selected target feature, primary target category, selected accuracy metric, selected bias metric, bias threshold, input variables, expected output.
    Output: This unit test will check if the function is giving the output as expected or not.
    '''   
    common_dataframe = pd.read_csv(datafile)
    common_dataframe[selected_target] = common_dataframe[selected_target].str.strip()
    data_prep = preprocess_data(data = common_dataframe[selected_features])

    predicted_data = pd.read_csv(predfile)
    predicted_data = predicted_data.values.reshape(1, -1)[0]

    X_test = data_prep
    common_dataframe.loc[common_dataframe[selected_target] == target_category, selected_target] = 1
    common_dataframe.loc[common_dataframe[selected_target] != 1, selected_target] = 0
    y_test = common_dataframe[selected_target].astype(int)
    y_pred = (predicted_data >= 0.5).astype(int)    

    feats_dict = {}
    for i in selected_features:
        print(i,selected_features)
        feats_dict[i] = [f'{i}_{x}' for x in common_dataframe[i].unique()]
    
    result1 = generate_dissimilarity_report(X_test, y_test, y_pred, acc_metrics = selected_acc_metrics, method = selected_bias_metrics, threshold = bias_threshold/100)    
    print(result1[result1['feature']==inputs[0]]['Flag'].values, expected_output[0])
    result2, figs = generateFairnessReport(X_test, y_test, y_pred, feats_dict)
    print(result2[result2['Feature']==inputs[1]]['FPR Parity'].values, expected_output[1])
    
    assert result1[result1['feature']==inputs[0]]['Flag'].values == expected_output[0] and result2[result2['Feature']==inputs[1]]['FPR Parity'].values == expected_output[1]
