from django.urls import path
from . import views

urlpatterns = [
    path('', views.robustness, name='gen'),
    path('getfeatures/', views.get_features, name='getfeatures'),
    path('pkgvulnerability_submit/', views.pkgvulnerability_submit, name='pkgvulnerability_submit')
]