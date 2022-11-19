# from gettext import dpgettext
from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, JsonResponse
from .forms import DataSetFile, UnstructuredDataFileForm
from django.views.decorators.csrf import csrf_exempt
from scipy.stats import chisquare
from .associatebias.forms import AssociationBiasEmbeddingModel

from matplotlib import pyplot as plt
import seaborn as sns

from fairnesstest.settings import BASE_DIR, TRANSPARENCY_URL
folder_path = BASE_DIR/'structureddata//static//structureddata'


def unstructured_data(request):
    if request.method=='POST':
        unstructureddata_form = UnstructuredDataFileForm(request.POST, request.FILES)
        if unstructureddata_form.is_valid():
            global common_dataset_file
            common_dataset_file = unstructureddata_form.cleaned_data["common_dataset_file"]
            common_dataset_file = save_file(common_dataset_file)
            AssociationBiasEmbeddingModel_form = AssociationBiasEmbeddingModel()
            bias_html = 1
            return render(request, 'unstructureddata/unstructured_data.html', {'unstructureddata_form':unstructureddata_form, 'bias_html':bias_html, 'AssociationBiasEmbeddingModel_form':AssociationBiasEmbeddingModel_form})
    else:
        unstructureddata_form = UnstructuredDataFileForm()
    bias_html = 0
    return render(request, 'unstructureddata/unstructured_data.html', {'unstructureddata_form':unstructureddata_form, 'bias_html':bias_html})


def save_file(file_object):
    file_name = str(file_object)
    file_name = 'common_dataset_file.csv'
    file_path = f'{folder_path}\\files\\{file_name}'
    with open(file_path, 'wb+') as f:
        for chunk in file_object.chunks():
            f.write(chunk)

    return file_path