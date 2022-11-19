from email.mime import image

import pandas as pd
import numpy as np
import pickle
import mpld3
import logging
import matplotlib.pyplot as plt

from django.shortcuts import render
from django.http import JsonResponse

# Create your views here.
from .forms import ShapFileForm, CAMFileForm, LimeFileForm, ProtodashForm
from django.views.decorators.csrf import csrf_exempt
from transparencytest.settings import BASE_DIR


folder_path = BASE_DIR/'rough//static//rough'

def transparency(request):
    if request.method == 'POST':
        pass
    else:
        shap_form = ShapFileForm()
        lime_form = LimeFileForm()
        cam_form = CAMFileForm()
        proto_form = ProtodashForm()
    return render(request, 'rough/transparency.html', {'shap_form': shap_form, 'lime_form': lime_form, 'cam_form': cam_form, 'proto_form': proto_form})


@csrf_exempt
def camsubmit(request):
    import subprocess, string, random
    if request.method == 'POST':
        response = {}
        status_code = 200
        response['status'] = 200
        try:
            model_file = request.FILES['mfile']
            image_file = request.FILES['file']

            # load keras model
            model_file_path = save_file(model_file)
            image_file_name = save_file(image_file)
            unique_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10))
            cmds = f"C:/Users/deepak.upman/Miniconda3/Scripts/activate\nconda activate app_tf2\npython {BASE_DIR}//rough//cam.py --model_file_path {model_file_path} --image_file_name {image_file_name} --folder_path {folder_path} --unique_id {unique_id}\n"
            cmd = subprocess.Popen("cmd.exe", universal_newlines=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            output, errors = cmd.communicate(cmds)
            cmd.wait()
            if not output.__contains__("Heat Map generated successfully!!!"):
                response['status'] = 500
            else:
                filenames = [f'{unique_id}_original.png', f'{unique_id}_heat_map.png', f'{unique_id}_cam_result.png']
                response['filenames'] = filenames
        except Exception as ex:
            status_code = 500
            response['status'] = 500
        return JsonResponse(response, status=status_code)
    else:
        form = CAMFileForm()
    return render(request, 'rough/cam.html', {'form': form})


@csrf_exempt
def proto_submit(request):
    if request.method == 'POST':
        response = {}
        status_code = 200
        response['status'] = 200
        try:
            csv_file = request.FILES['csv_file']
            m = request.POST['m']
            result = aix360_protodash(csv_file, int(m))
            response['result'] = result
        except Exception as ex:
            status_code = 500
            response['status'] = 500
        return JsonResponse(response, status=status_code)


@csrf_exempt
def limesubmit(request):
    if request.method == 'POST':
        response = {}
        response['status'] = 200
        status_code = 200
        try:
            model_file = request.FILES['mfile']
            train_file = request.FILES['file']
            test_file = request.FILES['testfile']
            selected_features = request.POST['selected_feature']
            test_input_idx = request.POST['test_input_row']
            num_features = request.POST['num_feature']
            exp = lime_tabularExplainer(
                model_file, train_file, test_file, selected_features, test_input_idx, num_features)
            response['exp'] = exp['exp']
            response['test_row'] = exp['test_row']

        except Exception as ex:
            status_code = 500
            response['status'] = 500
        return JsonResponse(response, status = status_code)
    else:
        form = LimeFileForm()
        return render(request, 'rough/lime.html', {'form': form})

@csrf_exempt
def shapsubmit(request):
    if request.method == 'POST':
        response = {}
        status_code = 200
        try:
            response['status'] = 200
            model_file = request.FILES['mfile']
            train_file = request.FILES['file']
            test_file = request.FILES['testfile']
            test_input_idx = request.POST['test_input_row']
            results = shap_kernelExplainer(
                model_file, train_file, test_file, test_input_idx)
            response['result'] = results['test_row']
            response['plot_image'] = results['plot_image']
        except Exception as ex:
            status_code = 500
            response['status'] = 500
        return JsonResponse(response, status = status_code)
    else:
        form = ShapFileForm()
        return render(request, 'rough/lime.html', {'form': form})

@csrf_exempt
def get_features(request):
    response = {}
    response['dataset_columns'] = []
    status_code = 200
    try:
        data = pd.read_csv(request.FILES['file'])
        response['dataset_columns'] = list(data.columns)
        response['status'] = 200
    except Exception as exp:
        status_code = 500
        response['status'] = 500
    return JsonResponse(response, status = status_code)


def aix360_protodash(csv_file=None, m=None):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from aix360.algorithms.protodash import ProtodashExplainer
    data = pd.read_csv(csv_file)
    cat_cols = list(set(data.columns).difference(data.describe().columns))
    if len(cat_cols) == 0:
        return "No Categorical Variable is present"
    else:
        data.loc[:, cat_cols] = SimpleImputer(
            missing_values=np.nan, strategy='most_frequent').fit_transform(data.loc[:, cat_cols])
        data_cat = data.loc[:, cat_cols]

        one_hot_encoder = OneHotEncoder(sparse=False)
        one_hot_encoded = one_hot_encoder.fit_transform(data_cat)

        explainer = ProtodashExplainer()

        (W, S, _) = explainer.explain(one_hot_encoded, one_hot_encoded, m=m)

        # Display the prototypes along with their computed weights
        inc_prototypes = data_cat.iloc[S, :].copy()
        # Compute normalized importance weights for prototypes
        inc_prototypes["Weights of Prototypes"] = np.around(W/np.sum(W), 2)
        inc_prototypes = inc_prototypes.reset_index(drop=True)
        return inc_prototypes.to_html()


def lime_tabularExplainer(model_file_path=None, train_data_file_path=None, test_data_file_path=None, target_variable_name=None, test_input_idx=None, num_features=None):
    from lime import lime_tabular
    if not train_data_file_path or not model_file_path or not test_data_file_path or not target_variable_name:
        return 0
    # read model from pickle file
    loaded_model = pickle.load(model_file_path)

    # read data from csv file
    X_train = pd.read_csv(train_data_file_path)
    X_test = pd.read_csv(test_data_file_path)

    class_name = X_train[target_variable_name].unique()

    X_train = X_train.drop(target_variable_name, axis=1)
    X_test = X_test.drop(target_variable_name, axis=1)

    # TODO: mode will be selected from webpage
    model_mode = "classification"

    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns,
        class_names=class_name,
        mode=model_mode
    )
    exp = explainer.explain_instance(
        data_row=X_test.iloc[int(test_input_idx)],
        predict_fn=loaded_model.predict_proba,
        num_features=int(num_features),
        top_labels=2#len(class_name)
    )
    exp = exp.as_html()
    return {'exp': exp, 'test_row': X_test.iloc[int(test_input_idx)].to_json()}


def shap_kernelExplainer(model_file_path=None, train_data_file_path=None, test_data_file_path=None, test_input_idx=None):
    import shap
    if not train_data_file_path or not model_file_path or not test_data_file_path:
        return 0

    # read model from pickle file
    loaded_model = pickle.load(model_file_path)

    # read data from csv file
    X_train = pd.read_csv(train_data_file_path)
    X_test = pd.read_csv(test_data_file_path)

    explainer = shap.KernelExplainer(loaded_model.predict_proba, X_train)
    shap_values = explainer.shap_values(X_test, nsamples=10)
    plt.clf()
    shap.force_plot(explainer.expected_value[0], shap_values[0]
                    [0, :], X_test.iloc[int(test_input_idx), :], matplotlib=True, show=False)
    fig = plt.gcf()
    plot_image = mpld3.fig_to_html(fig)
    #plt.savefig(folder_path/'images\\shap_plot_img.png',
    #            dpi=150, bbox_inches='tight')
    return {'plot_image': plot_image, 'test_row': X_test.iloc[int(test_input_idx)].to_json()}


def save_file(file_object):
    file_name = str(file_object)
    file_path = f'{folder_path}\\files\\{file_name}'
    with open(file_path, 'wb+') as f:
        for chunk in file_object.chunks():
            f.write(chunk)
    return file_path
