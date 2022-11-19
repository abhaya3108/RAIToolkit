from django import forms

class DataFileForm(forms.Form):
    train_file = forms.FileField(label='Upload Train File (e.g., csv)', error_messages={'required':'Please Upload Train File'}, widget=forms.FileInput(attrs={'class':'form-control', 'accept':".csv"}))

class CAMFileForm(forms.Form):
    model_file = forms.FileField(label='Upload Model File (e.g., h5)', error_messages={'required':'Please Upload Model File'}, widget=forms.FileInput(attrs={'class':'form-control', 'id':'cam_model_file'}))
    img_file = forms.ImageField(label="Upload Image File (e.g., png/jpg)", error_messages={'required':'Please Enter Class Idx'}, widget=forms.FileInput(attrs={'class':'form-control', 'id':'cam_img_file'}))
    #class_idx = forms.IntegerField(label='Enter Class Idx', error_messages={'required':'Please Enter Class Idx'}, widget=forms.NumberInput(attrs={'class':'form-control', 'id':'class_idx'}))

class LimeTextFileForm(forms.Form):
    Pipeline_file = forms.FileField(label='Upload Pipeline file (e.g., pickle)', error_messages={'required':'Please Upload Pipeline/Pickel File'}, widget=forms.FileInput(attrs={'class':'form-control', 'id':'limetext_model_file', 'accept':".pkl"}))
    train_file = forms.FileField(label='Upload Train File (e.g., csv)', error_messages={'required':'Please Upload Train File'}, widget=forms.FileInput(attrs={'class':'form-control', 'id':'limetext_train_file', 'accept':".csv"}))
    test_file = forms.FileField(label='Upload Test File (e.g., csv)', error_messages={'required':'Please Upload Test File'}, widget=forms.FileInput(attrs={'class':'form-control', 'id':'limetext_test_file', 'accept':".csv"}))
    test_input_row_no = forms.IntegerField(label='Enter Test Input Row Number', error_messages={'required':'Please Enter Test Input Row Number'}, widget=forms.NumberInput(attrs={'class':'form-control', 'id':'test_input_row_no1'}))