from django import forms

class DataSetFile(forms.Form):
   file = forms.FileField(required=True)

