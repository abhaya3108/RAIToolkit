# Libraries
import pytest
import os
from general.views import pkgvulnerability

# Robustness Dataset directory
TEST_DATASET_DIR = "../Test DataSets/Robustness/"

correct_req_path = os.path.join(TEST_DATASET_DIR,'/General/Package_Vulnerability/Set 1/')
incorrect_req_path = os.path.join(TEST_DATASET_DIR,'/General/Package_Vulnerability/Set 2/')

##### Robustness Unit test cases #####

@pytest.mark.general
@pytest.mark.major
@pytest.mark.parametrize("req_file_name,req_file_path,expected_output", 
                                                                        [('requirements_correct.txt', correct_req_path, True),
                                                                         ('requirements_incorrect.txt', incorrect_req_path, False)
                                                                        ])
def test_pkgvulnerability(req_file_name,req_file_path,expected_output):
    '''
    Purpose: This unit test checks pkgvulnerability module using correct and incorrect requirement files
    Input: Requirement file, Requirement file path and expected output 
    Output: Test will Pass if function output matches the expectation else Fail
    Note: general (to mark type of functionality) and major (to mark functionality impact) markers have been used. 
    '''    

    # Checking function output against expected output
    assert ('No known vulnerabilities found' in pkgvulnerability(req_file_name,req_file_path)) == expected_output