from django import forms

class ShapFileForm(forms.Form):
    model_file = forms.FileField(label='Upload Model File (e.g., pickle)', error_messages={'required':'Please Upload Model/Pickel File'}, widget=forms.FileInput(attrs={'class':'form-control', 'id':'shap_model_file', 'accept':".pkl"}))
    train_file = forms.FileField(label='Upload Train File (e.g., csv)', error_messages={'required':'Please Upload Train File'}, widget=forms.FileInput(attrs={'class':'form-control', 'id':'shap_train_file', 'accept':".csv"}))
    test_file = forms.FileField(label='Upload Test File (e.g., csv)', error_messages={'required':'Please Upload Test File'}, widget=forms.FileInput(attrs={'class':'form-control', 'id':'shap_test_file', 'accept':".csv"}))
    test_input_row_no = forms.IntegerField(label='Enter Test Input Row Number', error_messages={'required':'Please Enter Test Input Row Number'}, widget=forms.NumberInput(attrs={'class':'form-control', 'id':'shap_test_input_row_no'}))

class LimeFileForm(forms.Form):
    model_file = forms.FileField(label='Upload Model File (e.g., pickle)', error_messages={'required':'Please Upload Model/Pickel File'}, widget=forms.FileInput(attrs={'class':'form-control', 'id':'lime_model_file', 'accept':".pkl"}))
    train_file = forms.FileField(label='Upload Train File (e.g., csv)', error_messages={'required':'Please Upload Train File'}, widget=forms.FileInput(attrs={'class':'form-control', 'id':'lime_train_file', 'accept':".csv"}))
    test_file = forms.FileField(label='Upload Test File (e.g., csv)', error_messages={'required':'Please Upload Test File'}, widget=forms.FileInput(attrs={'class':'form-control', 'id':'lime_test_file', 'accept':".csv"}))
    test_input_row_no = forms.IntegerField(label='Enter Test Input Row Number', error_messages={'required':'Please Enter Test Input Row Number'}, widget=forms.NumberInput(attrs={'class':'form-control', 'id':'test_input_row_no'}))
    num_feature = forms.IntegerField(label='Enter Number of top feature', error_messages={'required':'Please Enter Number of feature'}, widget=forms.NumberInput(attrs={'class':'form-control', 'id':'num_feature'}), initial=6)

class DataFileForm(forms.Form):
    train_file = forms.FileField(label='Upload Train File (e.g., csv)', error_messages={'required':'Please Upload Train File'}, widget=forms.FileInput(attrs={'class':'form-control', 'accept':".csv"}))
    
class ProtodashForm(forms.Form):
    csv_file = forms.FileField(label='Upload Train File (e.g., csv)', error_messages={'required':'Please Upload Train File'}, widget=forms.FileInput(attrs={'class':'form-control', 'id':'proto_train_file', 'accept':".csv"}))
    m = forms.IntegerField(label='Enter Number of Rows', error_messages={'required':'Please Enter Number of rows'}, widget=forms.NumberInput(attrs={'class':'form-control', 'id':'m'}))