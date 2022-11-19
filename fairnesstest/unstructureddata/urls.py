from django.urls import path,include
from . import views
from .associatebias import views as associate_views

urlpatterns = [

    path('unstruct/', views.unstructured_data, name='unstructdata'),
    path('unstruct/associatebiasgetfeatures/', associate_views.associatebias_get_features, name='associatebiasgetfeatures'),
    path('unstruct/associatebiassubmit/', associate_views.associatebias_submit, name='associatebiassubmit'),
]