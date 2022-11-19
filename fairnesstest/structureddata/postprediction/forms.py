from django import forms

class LableBiasPostPredictionForm(forms.Form):
    # test_file = forms.FileField(label='Upload Test Data', error_messages={'required':'Please Upload Test Data'}, widget=forms.FileInput(attrs={'class':'form-control', 'accept':".csv"}))
    prediction_file = forms.FileField(label='Upload Predicted Probabilities', error_messages={'required':'Please Upload Predicted Probabilities'}, widget=forms.FileInput(attrs={'class':'form-control', 'accept':".csv"}))
