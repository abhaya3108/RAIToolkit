from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, JsonResponse
#from .forms import DataSetFile
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
        from ..views import common_dataset_file,common_dataframe
        #data = pd.read_csv(common_dataset_file)
        num_cols = common_dataframe.select_dtypes(exclude=['object', 'category']).columns #data.describe().columns
        cat_cols = common_dataframe.select_dtypes(include=['object', 'category']).columns #list(set(data.columns).difference(num_cols))
        globals()['dataset_file'] = preprocess_data(common_dataframe)
        response['categorical_dataset_columns'] = sorted(list(set(dataset_file.columns)-set(num_cols)))
        response['continuous_dataset_columns'] = list(num_cols)
        response['target_dataset_columns'] = list(dataset_file.columns)
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
        feat_catcol_found = []
        lambda_filter_func = lambda col: col if feat_col in col else None
        for feat_col in protectedVar_categorical:
            feat_catcol_found = feat_catcol_found + list(filter(lambda_filter_func,data.columns)) #[(col if feat_col in col) for col in data.columns]
        feat_contcol_found = []
        for feat_col in protectedVar_continuous:
            feat_contcol_found = feat_contcol_found + list(filter(lambda_filter_func,data.columns))
        feat_tarcol_found = []
        for feat_col in targetVar:
            feat_tarcol_found = feat_tarcol_found + list(filter(lambda_filter_func,data.columns))
        # print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        # print(feat_catcol_found)
        # print(feat_contcol_found)
        # print(feat_tarcol_found)
        # print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        # labelbias_output = plotProbDist(protectedVar_categorical, protectedVar_continuous, targetVar, data)
        labelbias_output, label_bias_graphs, pre_pred_label_bias = plotProbDist(protectedVar_categorical, protectedVar_continuous, targetVar, data)
        # print('labelbias_output :',labelbias_output)
        # path = BASE_DIR/"structureddata/static/structureddata/label_bias_images"
        # folders = os.listdir(path)
        response['labelbias_output'] = labelbias_output
        response['user_selection'] = {'selected_categorical_features':protectedVar_categorical, 'selected_continuous_features':protectedVar_continuous, 'selected_target_features':targetVar}
        response['status'] = 200
        # print('dir :',folders)
        # response['listOfFiles'] = folders
        # print('os.path.splitext(x) :',os.path.splitext(x))
        # print('listOfFiles :',response)
        response['label_bias_graphs'] = label_bias_graphs 
        response['pre_pred_label_bias'] = pre_pred_label_bias
        return JsonResponse(response)