from django.urls import path
from . import views

urlpatterns = [
    path('', views.transparency, name='transparency'),
    path('getfeatures', views.get_features, name='getfeatures'),
    path('limesubmit', views.limesubmit, name='limesubmit'),
    path('shapsubmit', views.shapsubmit, name='shapsubmit'),
    path('camsubmit', views.camsubmit, name='camsubmit'),
    path('protosubmit', views.proto_submit, name='protosubmit')
]