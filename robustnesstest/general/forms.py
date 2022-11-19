from django import forms

class PkgVulnerabilityForm(forms.Form):
    requirement_file = forms.FileField(label='Upload Requirement File (e.g., txt)', error_messages={'required':'Please Upload Requirement File'}, widget=forms.FileInput(attrs={'class':'form-control', 'id':'requirement_file', 'accept':".txt"}))

class DataFileForm(forms.Form):
    train_file = forms.FileField(label='Upload Train File (e.g., csv)', error_messages={'required':'Please Upload Train File'}, widget=forms.FileInput(attrs={'class':'form-control', 'accept':".csv"}))
