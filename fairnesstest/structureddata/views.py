# from gettext import dpgettext
from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, JsonResponse
from .forms import DataSetFile, StructuredDataFileForm
from django.views.decorators.csrf import csrf_exempt
from scipy.stats import chisquare
from .misc.preprocessing import preprocess_data
from .postpredictionlablebias.forms import LableBiasPostPredictionForm

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from fairnesstest.settings import BASE_DIR, TRANSPARENCY_URL, TRANSPARENCY_UNSTRUCT_IMG_URL, TRANSPARENCY_UNSTRUCT_TXT_URL, ROBUSTNESS_URL
folder_path = BASE_DIR/'structureddata//static//structureddata'


def home_page(request):
    return render(request, 'structureddata/initial_page.html', {'TRANSPARENCY_URL': TRANSPARENCY_URL,'TRANSPARENCY_UNSTRUCT_IMG_URL': TRANSPARENCY_UNSTRUCT_IMG_URL, 'TRANSPARENCY_UNSTRUCT_TXT_URL': TRANSPARENCY_UNSTRUCT_TXT_URL,'ROBUSTNESS_URL':ROBUSTNESS_URL})

def structured_data(request):
    if request.method=='POST':
        structureddata_form = StructuredDataFileForm(request.POST, request.FILES)
        if structureddata_form.is_valid():
            global common_dataset_file
            common_dataset_file = structureddata_form.cleaned_data["common_dataset_file"]
            common_dataset_file = save_file(common_dataset_file)
            #Shagun Kala: Added common_dataframe variable to reuse in all tests. 6th July, 2022. 
            global common_dataframe
            if common_dataset_file.endswith('.csv'):
                common_dataframe = pd.read_csv(common_dataset_file)
            elif common_dataset_file.endswith('.xlsx'):
                common_dataframe = pd.read_excel(common_dataset_file)
            print('Data imported and read successfully!')
            lableBiasPostPrediction_form = LableBiasPostPredictionForm()
            bias_html = 1
            return render(request, 'structureddata/structured_data.html', {'structureddata_form':structureddata_form, 'bias_html':bias_html, 'lableBiasPostPrediction_form':lableBiasPostPrediction_form})
    else:
        structureddata_form = StructuredDataFileForm()
    bias_html = 0
    return render(request, 'structureddata/structured_data.html', {'structureddata_form':structureddata_form, 'bias_html':bias_html})


# dataset_file = None
# import pandas as pd
# @csrf_exempt
# def samplebias_getfeatures(request):
#     # if request.method == 'GET':
#     #     response = {}
#     #     response['dataset_columns'] = []
#     #     try:
#     #         file_form = DataSetFile(request.POST, request.FILES)
#     #         if(file_form.is_valid()):
#     #             file = file_form.cleaned_data["file"]
#     #             data = pd.read_csv(file)
#     #             globals()['dataset_file'] = preprocess_data(data)
#     #             response['dataset_columns'] = list(dataset_file.columns)
#     #             response['status'] = 200
#     #     except Exception as exp:
#     #         response['status'] = 300
#     # return JsonResponse(response)
    
#     response = {}
#     response['dataset_columns'] = []
#     try:
#         data = pd.read_csv(common_dataset_file)
#         globals()['dataset_file'] = preprocess_data(data)
#         response['dataset_columns'] = list(dataset_file.columns)
#         response['status'] = 200
#     except Exception as exp:
#         response['status'] = 300
#     return JsonResponse(response)


# @csrf_exempt
# def samplebias_submit(request):
#     '''
#     This will generate the sample bias output on user selections
#     '''
#     if request.method == 'POST':
#         response = {}
#         samplebias_output = []
#         print("request.body :",request.body)
#         selected_features = request.body.decode('utf-8').split(',')
#         # data_prep = preprocess_data(data = dataset_file)
#         data_prep = dataset_file
#         features_of_interest = selected_features

#         def test_sample_bias(data, column):
#             '''
#             Column must be categorical. If numerical column, bin the values first else this will not work.
#             '''
#             return 1 - chisquare(f_obs = data[column])[1]

#         #Creating dict with feature name and probability value
#         feature_probability_vals = dict([(col, test_sample_bias(data_prep, col)) for col in features_of_interest])

        #This section can be removed once text is replaced by graph
        #samplebias_output = [f'For {col}: {round(val, 2)} ' for col, val in feature_probability_vals.items()]
      
#         '''for col in features_of_interest:
#             # print(f'Confidence in favour of sample bias for {col} = {test_sample_bias(data_prep, col)}')
#             samplebias_output.append(f'Confidence in favour of sample bias for {col} = {test_sample_bias(data_prep, col)}')'''
        
#         _ = sample_bias_graph(feature_probability_vals)

#         response['samplebias_output'] = samplebias_output
#         response['selected_features'] = selected_features
#         response['status'] = 200
#         return JsonResponse(response)

# def sample_bias_graph(feature_probability_vals:dict):
#     '''This function generates takes dictionary, with key as feature name and value as probability, and creates bar chart.
#     '''
#     fig, ax = plt.subplots(constrained_layout=True)
#     fig.set_figheight(4)
    
#     #Make figure width dynamic based on number of parameters
#     num_feat = len(feature_probability_vals)
#     fig_width = num_feat * 2
#     if fig_width < 3:
#         #Set minimum width
#         fig_width = 3
#     elif fig_width > 12:
#         #Set maximum width
#         fig_width = 12
#     else:
#         pass
    
#     fig.set_figwidth(fig_width)

#     x_axis = range(0, num_feat)
#     x_labels = list(feature_probability_vals.keys())
#     ax.bar(x_axis, list(feature_probability_vals.values()), width=0.2, color='purple')
#     ax.set_title('Confidence Score Chart')
#     ax.set_ylim(0,1)
#     ax.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
#     ax.set_xticks(list(range(0,num_feat)))
#     ax.set_xticklabels(x_labels, rotation = 30, wrap=False)
#     ax.set_xlabel(r'${Features} \longrightarrow$')
#     ax.set_ylabel(r'${Confidence Score} \longrightarrow$')
#     plt.savefig(BASE_DIR/'structureddata/static/structureddata/images/sample_bias_images/sample_bias.png', dpi=150)
#     return None



def save_file(file_object):
    file_name = str(file_object)
    file_name = 'common_dataset_file.csv'
    file_path = f'{folder_path}\\files\\{file_name}'
    with open(file_path, 'wb+') as f:
        for chunk in file_object.chunks():
            f.write(chunk)

    return file_path