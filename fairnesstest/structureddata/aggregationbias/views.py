from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .simpsons_paradox import SimpsonsParadox
from fairnesstest.settings import BASE_DIR

import json
import pandas as pd


dataset_file = None


@csrf_exempt
def aggregationbias_get_features(request):
    status_code = 200
    response = {}
    response['categorical_dataset_columns'] = []
    try:
        from ..views import common_dataset_file
        data = pd.read_csv(common_dataset_file)
        globals()['dataset_file'] = data#preprocess_data(data)
        response['dataset_columns'] = list(dataset_file.columns)
        response['status'] = 200
    except Exception as exp:
        response['status'] = 300
        status_code = 300
    return JsonResponse(response, status=status_code)


@csrf_exempt
def aggregationbias_submit(request):
    '''
    This will generate the aggregation bias output on user selections
    '''
    if request.method == 'POST':
        response = {}
        request_data = json.loads(request.body)
        data = dataset_file
        params = {
            "df": data,
            "dv": request_data['dependent_variable'],
            "ignore_columns": request_data['ignore_columns'],
            "bin_columns": request_data['bin_columns'],
            'standardize': bool(request_data['standardize']),
            'bin_method': request_data['bin_method'],  
            'weighting': bool(request_data['weighting']), 
            'max_pvalue': float(request_data['max_P_value']), 
            'min_coeff': float(request_data['min_coeff']),
            'min_corr': float(request_data['min_corr']),
            'output_plots': True,
        }
        try:
            sp = SimpsonsParadox(**params)
            output = sp.get_simpsons_pairs()
            status_code = 200
        except Exception as exp:
            if (exp.args[0] == 'You have a non-binary DV. Pass a value to the target_category in the function or re-bin your DV prior to using the function.'):
                params['target_category'] = 1
                sp = SimpsonsParadox(**params)
                output = sp.get_simpsons_pairs()
                status_code = 200
            else:
                output = "ERROR: Unable to process the request"
                status_code = 500
        response['output'] = output
        response['status'] = status_code
        return JsonResponse(response, status=status_code)
