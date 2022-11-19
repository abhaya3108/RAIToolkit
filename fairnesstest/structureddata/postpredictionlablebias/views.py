from django.shortcuts import render, redirect
from .forms import LableBiasPostPredictionForm
from ..assessment.fairness_analysis import generate_dissimilarity_report, generateFairnessReport
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from ..misc.preprocessing import preprocess_data
import json
import numpy as np
from sklearn.impute import SimpleImputer

@csrf_exempt
def lablebias_postPrediction(request):
    if request.method=='POST':
        response = {}
        form = LableBiasPostPredictionForm(request.POST, request.FILES)
        if form.is_valid():        
            ##Reading probabilities file
            prediction_file = form.cleaned_data["prediction_file"]
            global predicted_data

            #Predictions File Validity Check #1
            try:
                predicted_data = pd.read_csv(prediction_file)
            except:
                response['status'] = 210
                return JsonResponse(response)

            #Predictions File Validity Check #2
            if predicted_data.shape[1] != 1:
                response['status'] = 220
                return JsonResponse(response)
            else:
                pass

            predicted_data = predicted_data.values.reshape(1, -1)[0]

            #Predictions File Validity Check #3
            for i in predicted_data:
                if i > 1.0 or i < 0:
                    response['status'] = 280
                    return JsonResponse(response)
                else:
                    pass

            global common_dataframe  
            from ..views import common_dataframe
            ##Processing common input file         
            # common_dataframe = common_dataframe.select_dtypes(include=['object', 'category'])
            
            # #Dataset File Validity Check #1
            # if len(common_dataframe.columns) <= 1:
            #     response['status'] = 240
            #     return JsonResponse(response)
            # else:
            #     pass

            # common_dataframe[:] = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent').fit_transform(common_dataframe)
            # common_dataframe[:] = SimpleImputer(missing_values = None, strategy = 'most_frequent').fit_transform(common_dataframe)
            common_dataframe = common_dataframe.fillna(common_dataframe.mode().iloc[0])
            # data_prep = preprocess_data(data = common_dataframe)#test_data)
            # num_cols = common_dataframe.select_dtypes(exclude=['object', 'category']).columns
            response['test_data_columns'] = sorted(list(set(common_dataframe.columns)))
            response['target_columns'] = list(common_dataframe.columns)

            print(common_dataframe.shape[0], predicted_data.shape[0])

            #Predictions File Validity Check #4
            if common_dataframe.shape[0] == predicted_data.shape[0]:
                response['status'] = 200
            else:
                response['status'] = 250

            return JsonResponse(response)

    elif request.method=='GET':
        response = {}
        
        try:
            if request.method == 'GET':
                target = list(request.GET.keys())[0]
                target_set = common_dataframe[target].astype(str).unique()
                response['target_set'] = list(set(target_set))
                response['status'] = 200
            else:
                response['status'] = 200

        except Exception as exp:
            print(exp)
            response['status'] = 300

        return JsonResponse(response)

    else:
        form = LableBiasPostPredictionForm()
    return render(request, 'structureddata/one.html', {'form': form})


@csrf_exempt
def generate_dissimilarity_report_postprediction(request):
    if request.method=='POST':
        response= {}
        
        data = json.loads(request.body)
        selected_features = data['selected_features']
        selected_target = data['selected_target']
        selected_acc_metrics = data['selected_acc_metrics']
        target_category = data['target_category']
        selected_bias_metrics = data['selected_bias_metrics']
        bias_threshold = float(data['bias_threshold'])
        num_flag = data['num_flag']
        print(selected_features, target_category, selected_bias_metrics, bias_threshold, num_flag)
        
        #Numerical columns check

        num_cols = common_dataframe[selected_features].select_dtypes(exclude=['object', 'category']).columns
        if num_flag == 'True':
            
            for i in num_cols:
                if common_dataframe[i].nunique() > 10:
                    response['status'] = 380
                    return JsonResponse(response)
                else:
                    pass
        else:
            pass

        #Changing to correct datatypes
        for i in num_cols:
            common_dataframe[i] = common_dataframe[i].astype(str)

        global data_prep
        data_prep = preprocess_data(data = common_dataframe[selected_features])

        global X_test
        X_test = data_prep
        global y_test
        data_file = common_dataframe.copy()
        data_file.loc[data_file[selected_target] == target_category, selected_target] = 1
        data_file.loc[data_file[selected_target] != 1, selected_target] = 0
        y_test = data_file[selected_target].astype(int)
        
        # y_pred = (y_pred_proba >= 0.5).astype(int)
        y_pred = (predicted_data >= 0.5).astype(int)

        feats_dict = {}
        for i in selected_features:
            feats_dict[i] = [f'{i}_{x}' for x in data_file[i].unique()]
        
        print(feats_dict)

        del data_file

        try:

            result1 = generate_dissimilarity_report(X_test, y_test, y_pred, acc_metrics = selected_acc_metrics, method = selected_bias_metrics,
                                threshold = bias_threshold/100)

            result2, figs = generateFairnessReport(X_test, y_test, y_pred, feats_dict)
            
            result1.fillna("NA", inplace = True)
            result2.fillna("NA", inplace = True)

            response['result1'] = '''<style>
                            .df th { background-color: #a100ff; color: white;}
                            table {
                            width: 50%;
                            }
                            </style>''' + result1.to_html(border = 2, justify = 'left', classes=['table table-stripped','df'])

            response['result2'] = '''<style>
                            .df th { background-color: #a100ff; color: white;}
                            table {
                            width: 50%;
                            }
                            </style>''' + result2.to_html(border = 2, justify = 'left', classes=['table table-stripped','df'])
            response['html_postpredbias_fig'] = "<br>".join(figs)
            response['user_selection'] = {'selected_features':", ".join(selected_features), 'selected_target':selected_target, 'selected_acc_metrics':selected_acc_metrics
                                            , 'selected_bias_metrics':selected_bias_metrics, 'bias_threshold':bias_threshold/100}
            response['status'] = 200
            return JsonResponse(response)

        except Exception as exp:
            print(exp)
            response['status'] = 300
            return JsonResponse(response)
    else:
        return render('lablebiaspostprediction')