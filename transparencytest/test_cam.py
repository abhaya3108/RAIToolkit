# Libraries
import pytest
import datetime as dt
import os
from unstructureddata.views import start_process
from pathlib import Path


# Transparency Dataset directory
TEST_DATASET_DIR = Path(__file__).resolve(
).parent.parent/"Test_DataSets\\Transparency\\"

model_file_path_pass = os.path.join(
    TEST_DATASET_DIR, "Unstructured\\CAM\\SET1\\resnet50.h5")

image_file_name_pass = os.path.join(
    TEST_DATASET_DIR, "Unstructured\\CAM\\SET1\\dog.png")

model_file_path_fail = os.path.join(
    TEST_DATASET_DIR, "Unstructured\\CAM\\SET1\\rresnet50.h5")
image_file_name_fail = os.path.join(
    TEST_DATASET_DIR, "Unstructured\\CAM\\SET1\\ddog.png")

##### Transparency Unit test cases #####


@pytest.mark.general
@pytest.mark.major
@pytest.mark.parametrize("model_file_path, image_file_name, expected_output",
                         [(model_file_path_pass, image_file_name_pass, True),
                          (model_file_path_fail,
                           image_file_name_fail, False)
                          ])
def test_start_process(model_file_path, image_file_name, expected_output):
    '''
    Purpose: This unit test checks start_process module using correct and incorrect requirement files
    Input: model file, image file path and expected output 
    Output: Test will Pass if function output matches the expectation else Fail
    Note: general (to mark type of functionality) and major (to mark functionality impact) markers have been used. 
    '''

    # Checking function output against expected output
    unique_id = dt.datetime.now().strftime("%m-%d-%y_%H-%M-%S")
    #''.join(random.choices(string.ascii_uppercase + string.digits, k = 10))
    output = start_process(model_file_path, image_file_name, unique_id)
    # if not output.__contains__(""):
    assert (output.__contains__(
        "Heat Map generated successfully!!!")) == expected_output
