from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, JsonResponse
from .forms import DataSetFile
from django.views.decorators.csrf import csrf_exempt
from scipy.stats import chisquare
from ..assessment.dist_analysis import plotProbDist
from ..misc.preprocessing import preprocess_data
import pandas as pd
from fairnesstest.settings import BASE_DIR
import os
import json

# categorical_dataset_file = None
dataset_file = None
@csrf_exempt
def labelbias_get_features(request):
    # if request.method == 'POST':
    #     response = {}
    #     response['categorical_dataset_columns'] = []
    #     try:
    #         file_form = DataSetFile(request.POST, request.FILES)
    #         if(file_form.is_valid()):
    #             file = file_form.cleaned_data["file"]
    #             data = pd.read_csv(file)
    #             globals()['dataset_file'] = preprocess_data(data)
    #             response['categorical_dataset_columns'] = list(dataset_file.columns)
    #             response['status'] = 200
    #     except Exception as exp:
    #         response['status'] = 300
    # return JsonResponse(response)
    response = {}
    response['categorical_dataset_columns'] = []
    try:
        from ..views import common_dataset_file
        data = pd.read_csv(common_dataset_file)
        globals()['dataset_file'] = preprocess_data(data)
        response['categorical_dataset_columns'] = list(dataset_file.columns)
        response['status'] = 200
    except Exception as exp:
        response['status'] = 300
    return JsonResponse(response)

@csrf_exempt
def labelbias_submit(request):
    '''
    This will generate the sample bias output on user selections
    '''
    for name in os.listdir('structureddata\\static\\structureddata\\label_bias_images'):
        os.remove('structureddata\\static\\structureddata\\label_bias_images\\' + name)
    if request.method == 'POST':
        response = {}
        labelbias_output = None
        data1 = json.loads(request.body)
        protectedVar_categorical = data1['selected_categorical_features']
        protectedVar_continuous = data1['selected_continuous_features']
        targetVar = data1['selected_target_features']
        data = dataset_file
        labelbias_output = plotProbDist(protectedVar_categorical, protectedVar_continuous, targetVar, data)
        print('labelbias_output :',labelbias_output)
        path = BASE_DIR/"structureddata/static/structureddata/label_bias_images"
        folders = os.listdir(path)
        response['labelbias_output'] = labelbias_output
        response['user_selection'] = {'selected_categorical_features':protectedVar_categorical, 'selected_continuous_features':protectedVar_continuous, 'selected_target_features':targetVar}
        response['status'] = 200
        print('dir :',folders)
        response['listOfFiles'] = folders
        # print('os.path.splitext(x) :',os.path.splitext(x))
        print('listOfFiles :',response)
        return JsonResponse(response)