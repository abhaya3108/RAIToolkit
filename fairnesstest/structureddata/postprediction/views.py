from django.shortcuts import render, redirect
from .forms import LableBiasPostPredictionForm
from ..assessment.fairness_analysis import generate_dissimilarity_report, generateFairnessReport
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def lablebias_postPrediction(request):
    if request.method=='POST':
        response = {}
        form = LableBiasPostPredictionForm(request.POST, request.FILES)
        if form.is_valid():        
            # test_file = form.cleaned_data["test_file"]
            from ..views import common_dataset_file
            global test_data
            test_data = pd.read_csv(common_dataset_file)

            prediction_file = form.cleaned_data["prediction_file"]
            global predicted_data
            predicted_data = pd.read_csv(prediction_file)
            predicted_data = predicted_data.values.reshape(1, -1)[0]

            from ..misc.preprocessing import preprocess_data
            global data_prep
            data_prep = preprocess_data(data = test_data)
            response['test_data_columns'] = list(data_prep.columns)

            response['status'] = 200
            return JsonResponse(response)
    else:
        form = LableBiasPostPredictionForm()
    return render(request, 'structureddata/one.html', {'form': form})


@csrf_exempt
def generate_dissimilarity_report_postprediction(request):
    if request.method=='POST':
        response= {}
        import json
        data = json.loads(request.body)
        selected_features = data['selected_features']
        selected_target = data['selected_target']
        selected_acc_metrics = data['selected_acc_metrics']

        # train_data = data_prep.sample(frac = 0.75, random_state = 200)
        # X_train = train_data.drop('class_ >50K', axis = 1)
        # y_train = train_data.loc[:, 'class_ >50K']

        # test_data = data_prep.loc[~data_prep.index.isin(train_data.index), :]
        global X_test
        X_test = data_prep.drop(selected_target, axis = 1) #targeted variables = adult file colums 'class_ >50K'
        global y_test
        y_test = data_prep.loc[:, selected_target]

        # from sklearn.ensemble import RandomForestClassifier
        # clf = RandomForestClassifier(random_state = 200)
        # clf.fit(X_train.values, y_train.values)
        # y_pred_proba = clf.predict_proba(X_test.values)[:, 1]

        # y_pred = (y_pred_proba >= 0.5).astype(int)
        y_pred = (predicted_data >= 0.5).astype(int)
        onehot_protected_attributes = selected_features #feature selection step2
        # onehot_protected_attributes = ['sex_ Male',
        #                        'race_ Amer-Indian-Eskimo',
        #                        'race_ Asian-Pac-Islander',
        #                        'race_ Black',
        #                        'race_ Other',
        #                        'race_ White']

        result1 = generate_dissimilarity_report(X_test, y_test, y_pred, onehot_protected_attributes, acc_metrics = selected_acc_metrics,
                              threshold = 0.1)

        result2 = generateFairnessReport(X_test, y_test, y_pred, onehot_protected_attributes)
        
        response['result1'] = result1.to_html()
        response['result2'] = result2.to_html()
        response['user_selection'] = {'selected_features':selected_features, 'selected_target':selected_target, 'selected_acc_metrics':selected_acc_metrics}
        response['status'] = 200
        return JsonResponse(response)
    else:
        return render('lablebiaspostprediction')