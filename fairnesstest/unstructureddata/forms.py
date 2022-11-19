from django import forms

class DataSetFile(forms.Form):
   file = forms.FileField(required=True)


class UnstructuredDataFileForm(forms.Form):
   common_dataset_file = forms.FileField(label='Upload Data Set File', error_messages={'required':'Please Upload Data Set File'}, widget=forms.FileInput(attrs={'class':'form-control', 'accept':".csv"}))