from django import forms
   
class DataSetFile(forms.Form):
   file = forms.FileField(required=True)   

class AssociationBiasEmbeddingModel(forms.Form):
    prediction_file = forms.FileField(label='Upload Embeddings File (*.model)', error_messages={'required':'Please upload Embeddings file'}, widget=forms.FileInput(attrs={'class':'form-control', 'accept':".model"}))

