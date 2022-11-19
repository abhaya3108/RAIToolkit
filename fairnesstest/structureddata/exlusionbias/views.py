from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, JsonResponse
from .forms import DataSetFile
from django.views.decorators.csrf import csrf_exempt
from scipy.stats import chisquare
from ..assessment.corr_analysis import checkCorr, plotCorrMap
from ..misc.preprocessing import preprocess_data
import pandas as pd
from fairnesstest.settings import BASE_DIR
import os
import json


# categorical_dataset_file = None
dataset_file = None
@csrf_exempt
def exclusionbias_get_features(request):
    # if request.method == 'POST':
    #     response = {}
    #     response['dataset_columns'] = []
    #     try:
    #         file_form = DataSetFile(request.POST, request.FILES)
    #         if(file_form.is_valid()):
    #             file = file_form.cleaned_data["file"]
    #             data = pd.read_csv(file)
    #             globals()['dataset_file'] = preprocess_data(data)
    #             response['dataset_columns'] = list(dataset_file.columns)
    #             response['status'] = 200
    #     except Exception as exp:
    #         response['status'] = 300
    # return JsonResponse(response)
    response = {}
    response['dataset_columns'] = []
    try:
        from ..views import common_dataset_file
        data = pd.read_csv(common_dataset_file)
        globals()['dataset_file'] = preprocess_data(data)
        response['dataset_columns'] = list(dataset_file.columns)
        response['status'] = 200
    except Exception as exp:
        response['status'] = 300
    return JsonResponse(response)

@csrf_exempt
def exclusionbias_submit(request):
    '''
    This will generate the sample bias output on user selections
    '''
    # for name in os.listdir('static\\label_bias_images'):
    #     os.remove('static\\label_bias_images\\' + name)
    if request.method == 'POST':
        response = {}
        try:
            for name in os.listdir('structureddata\\static\\structureddata\\exclusion_bias_images'):
                os.remove('structureddata\\static\\structureddata\\exclusion_bias_images\\' + name)
        except:
            pass
        
        exclusionbias_output = None
        data1 = json.loads(request.body)
        print("exclusion bias data from front end :",data1)
        protectedVar_features = data1['selected_protected_features']
        target_feature = data1['selected_target_features']
        cat_cat_features = data1['selected_radio_features']
        cont_cat_features = data1['selected_radio_1_features']
        cont_cont_features = data1['selected_radio_2_features']
        print("protectedVar_features :",protectedVar_features)
        print("target_feature :",target_feature)
        print("cat_cat_features :",cat_cat_features)
        print("cont_cat_features :",cont_cat_features)
        print("cont_cont_features :",cont_cont_features)

        for col in protectedVar_features:
            statements, _, _ =checkCorr(columnName_protectedVar = col, columName_targetVar = target_feature, df = dataset_file, cat_cat = cat_cat_features, cont_cat = cont_cat_features)

        heatmap_made, heatmap_statements = plotCorrMap(ColumnNameList_protectedVar = protectedVar_features, 
            df = dataset_file,
            cont_cat = cont_cat_features, 
            cat_cat = cat_cat_features, 
            cont_cont = cont_cont_features)

        print(heatmap_made, heatmap_statements)
        path = BASE_DIR/"structureddata/static/structureddata/exclusion_bias_images"
        folders = os.listdir(path)
        response['listOfFiles'] = folders
        response['exclusionbias_output'] = statements
        response['user_selection'] = {'selected_protected_features':protectedVar_features, 'selected_target_features':target_feature, 'selected_radio_features':cat_cat_features, 'selected_radio_1_features':cont_cat_features, 'selected_radio_2_features':cont_cont_features}
        response['status'] = 200
        # print('dir :',folders)
        # response['listOfFiles'] = folders
        # # print('os.path.splitext(x) :',os.path.splitext(x))
        # print('listOfFiles :',response)
        return JsonResponse(response)