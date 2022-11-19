from django.urls import path
from . import views

urlpatterns = [
    path('image/', views.transparency_image, name='image'),
    path('text/', views.transparency_text, name='text'),
    path('image/getfeatures/', views.get_features, name='getfeatures'),
    path('image/camsubmit/', views.camsubmit, name='camsubmit'),
    path('text/getfeatures/', views.get_features, name='getfeatures'),
    path('text/limetextsubmit/', views.limetextsubmit, name='limetextsubmit'),
]