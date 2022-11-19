from email.mime import image

import pandas as pd
import numpy as np
import pickle

from django.shortcuts import render
from django.http import JsonResponse

# Create your views here.
from .forms import CAMFileForm,LimeTextFileForm
from django.views.decorators.csrf import csrf_exempt
from transparencytest.settings import BASE_DIR


folder_path = BASE_DIR/'unstructureddata//static//unstructureddata'

def transparency_image(request):
    if request.method == 'POST':
        pass
    else:
        cam_form = CAMFileForm()
    return render(request, 'unstructureddata/transparency_image.html', {'cam_form': cam_form})

def transparency_text(request):
    if request.method == 'POST':
        pass
    else:
        limetext_form = LimeTextFileForm()
    return render(request, 'unstructureddata/transparency_text.html', {'limetext_form':limetext_form})

def start_process(model_file_path, image_file_name, unique_id):
    import subprocess, os
    
    cmds = os.environ.get('_CONDA_ROOT', "C:/Users/deepak.upman/Miniconda3/") + f"Scripts\\activate\nconda activate app_tf2\npython {BASE_DIR}//unstructureddata//cam.py --model_file_path {model_file_path} --image_file_name {image_file_name} --folder_path {folder_path} --unique_id {unique_id}\n"
    cmd = subprocess.Popen("cmd.exe", universal_newlines=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, errors = cmd.communicate(cmds)
    cmd.wait()
    return output

@csrf_exempt
def camsubmit(request):
    import os, glob
    from datetime import timedelta, datetime as dt
    if request.method == 'POST':
        response = {}
        try:
            dir_to_clean = f"{folder_path}/cam_images/"
            list_of_files = filter(os.path.isfile, glob.glob(dir_to_clean + '*'))
            for filename in list_of_files:
                print(filename)
                datetime_obj = dt.strptime("_".join(filename.split("\\")[-1].split(".")[0].split("_")[:2]), "%m-%d-%y_%H-%M-%S")
                yesterday = dt.now() - timedelta(days = 1 )
                if (datetime_obj <= yesterday):
                    os.remove(filename)
        except:
            print("Unable to delete files....")
            pass
        try:
            model_file = request.FILES['mfile']
            image_file = request.FILES['file']

            # load keras model
            model_file_path = save_file(model_file)
            image_file_name = save_file(image_file)
            unique_id = dt.now().strftime("%m-%d-%y_%H-%M-%S")
            output = start_process(model_file_path, image_file_name, unique_id)
            if not output.__contains__("Heat Map generated successfully!!!"):
                status_code = 500
                response['status'] = 500
            else:
                filenames = [f'{unique_id}_original.png', f'{unique_id}_heat_map.png', f'{unique_id}_cam_result.png']
                response['filenames'] = filenames
                status_code = 200
                response['status'] = 200
        except Exception as ex:
            status_code = 500
            response['status'] = 500
        return JsonResponse(response, status=status_code)
    else:
        form = CAMFileForm()
    return render(request, 'unstructureddata/cam.html', {'form': form})

@csrf_exempt
def limetextsubmit(request):
    if request.method == 'POST':
        response = {}
        response['status'] = 200
        try:
            Pipeline_file = request.FILES['mfile']
            train_file = request.FILES['file']
            test_file = request.FILES['testfile']
            selected_features = request.POST['selected_feature']
            test_input_idx = request.POST['test_input_row']
            Pipeline_file_path = save_file(Pipeline_file)
            exp = lime_textExplainer(Pipeline_file_path,train_file, test_file,selected_features,test_input_idx)
            response['exp'] = exp['exp']
            response['test_row'] = exp['test_row']       
        except Exception as ex:
            print(ex)
            response['status'] = 300
        return JsonResponse(response)
    else:
        form = LimeTextFileForm()
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
    

def lime_textExplainer(Pipeline_file_path = None,
                       train_data_file_path=None, test_data_file_path=None, 
                       target_variable_name=None, test_input_idx = None, num_features=None):
    
    from lime import lime_text
    #from sklearn.pipeline import make_pipeline
    if not train_data_file_path or not Pipeline_file_path or not test_data_file_path or not target_variable_name :
        return 0
    # read model from pickle file
    c = pickle.load(open(Pipeline_file_path, 'rb'))
    
    # read data from csv file
    X_train = pd.read_csv(train_data_file_path)
    X_test = pd.read_csv(test_data_file_path)

    #class_name = X_train[target_variable_name].unique()
    class_name = X_train[target_variable_name].unique()
    X_train = X_train.drop(target_variable_name, axis=1)
    print(X_test)
    
    #X_test= X_test.iloc[int(test_input_idx)]
    np_X_test = X_test.to_numpy()
    
    explainer = lime_text.LimeTextExplainer(
        class_names=class_name,
        )
    
    exp = explainer.explain_instance(np_X_test.item(int(test_input_idx)), c.predict_proba)
    predict_prob = exp.__dict__['predict_proba'][0]
    exp = exp.as_html()
    return {'exp':exp,'Document id:':test_input_idx ,'test_row':np_X_test.item(int(test_input_idx)), 'Probability':predict_prob}



def save_file(file_object):
    file_name = str(file_object)
    file_path = f'{folder_path}\\files\\{file_name}'
    with open(file_path, 'wb+') as f:
        for chunk in file_object.chunks():
            f.write(chunk)
    return file_path
