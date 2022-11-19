from email.mime import image

import os
import warnings
import subprocess
import pandas as pd
import numpy as np
import pickle
import mpld3
import logging
import matplotlib.pyplot as plt

from django.shortcuts import render
from django.http import JsonResponse

# Create your views here.
from .forms import PkgVulnerabilityForm
from django.views.decorators.csrf import csrf_exempt
from robustnesstest.settings import BASE_DIR
from log_exception import log_details
import traceback
import json
import requests

folder_path = BASE_DIR/'general//static//general'
requirement_file_path = f'{folder_path}\\files'

CONFIG_FILE = "config.json"
config = json.loads(open(CONFIG_FILE).read())

def robustness(request):
    if request.method == 'POST':
        pass
    else:
        pkg_form = PkgVulnerabilityForm()
    return render(request, 'general/robustness.html', {'pkg_form': pkg_form})

@csrf_exempt
def pkgvulnerability_submit(request):
        
    if request.method == 'POST':
        response = {}
        status_code = 200
        response['status'] = 200
        try:
            text_file = request.FILES['text_file']
            if str(text_file).rsplit('.')[-1] != 'txt':
                response['status'] = 320
                return JsonResponse(response, status=status_code)
            else:
                pass
            save_file(text_file)
            result = pkgvulnerability(text_file)  
            response['result'] = result
        except Exception as e:
            exception_type, filename, line_number = log_details(e)
            # print(traceback.format_exc())
            req = {"status": True,
                  "service_name": "Robustness",
                  "error_code": 400,
                  "exception_type": str(exception_type),
                  "file_name": filename,
                  "line_number": line_number,
                  "error_info": "NA"
                }

            resp = requests.post(config['error_handler_endpoint_url'], data=json.dumps(req), headers={"Content-Type": "application/json"})
            response['status'] = 500
        return JsonResponse(response, status=status_code)
 
@csrf_exempt 
def pkgvulnerability(text_file, file_path = requirement_file_path):
    ''' 
    This function checks package vulnerability from requirement file 
    '''
    
    os.chdir(requirement_file_path)
    audit_result = ''
    audit_string = ''
    try:
        audit_result = subprocess.run(args = 'pip-audit -r ' + f'{text_file}',check=False, capture_output=True, text=True )
        if 'No known vulnerabilities found' in audit_result.stderr:
            audit_string = 'Audit Completed! No known vulnerabilities found'
        elif audit_result.stdout == "":
            audit_string = 'Audit Cancelled! Invalid requirement file format found'
        else:
            audit_result.stdout = audit_result.stdout.replace("\n", "<br>")
            audit_string = f'Audit Completed! There are following vulnerabilities found during audit: <br>{audit_result.stdout}'
        return audit_string
    except subprocess.CalledProcessError:
        exception_type, filename, line_number = log_details(ex)
        # print(traceback.format_exc())
        req = {"status": True,
              "service_name": "Robustness",
              "error_code": 400,
              "exception_type": str(exception_type),
              "file_name": filename,
              "line_number": line_number,
              "error_info": "NA"
            }

        resp = requests.post(config['error_handler_endpoint_url'], data=json.dumps(req), headers={"Content-Type": "application/json"})
        return "Unknown issue while auditing requirement file"

@csrf_exempt
def get_features(request):
    response = {}
    response['dataset_columns'] = []
    status_code = 200
    try:
        data = pd.read_csv(request.FILES['file'])
        response['dataset_columns'] = list(data.columns)
        response['status'] = 200
    except Exception as e:
        exception_type, filename, line_number = log_details(e)
        # print(traceback.format_exc())
        req = {"status": True,
              "service_name": "Robustness",
              "error_code": 400,
              "exception_type": str(exception_type),
              "file_name": filename,
              "line_number": line_number,
              "error_info": "NA"
            }

        resp = requests.post(config['error_handler_endpoint_url'], data=json.dumps(req), headers={"Content-Type": "application/json"})
        status_code = 500
        response['status'] = 500
    return JsonResponse(response, status = status_code)
    
def save_file(file_object):
    file_name = str(file_object)
    file_path = f'{folder_path}\\files\\{file_name}'
    with open(file_path, 'wb+') as f:
        for chunk in file_object.chunks():
            f.write(chunk)
    return file_path