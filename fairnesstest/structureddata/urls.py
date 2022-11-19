from django.urls import path,include
from . import views
from .samplebias import views as samplebias_views
from .prepredictionlabelbias.views import labelbias_get_features,labelbias_submit
from .exlusionbias.views import exclusionbias_get_features,exclusionbias_submit
from .aggregationbias.views import aggregationbias_get_features, aggregationbias_submit
from .postpredictionlablebias import views as pp_views


urlpatterns = [
    path('', views.home_page, name='homepage'),
    path('fairness/struct/', views.structured_data, name='structdata'),

    path('fairness/samplebiasgetfeatures/', samplebias_views.samplebias_getfeatures, name='samplebiasgetfeatures'),
    path('fairness/samplebiassubmit/', samplebias_views.samplebias_submit, name='samplebiassubmit'),

    path('fairness/labelbiasgetfeatures/', labelbias_get_features, name='labelbiasgetfeatures'),
    path('fairness/labelbiassubmit/', labelbias_submit, name='labelbiassubmit'),

    path('fairness/exlusionbiasgetfeatures/', exclusionbias_get_features, name='exclusionbiasgetfeatures'),
    path('fairness/exclusionbiassubmit/', exclusionbias_submit, name='exclusionbiassubmit'),

    path('fairness/aggregationbiasgetfeatures/', aggregationbias_get_features, name='aggregationbiasgetfeatures'),
    path('fairness/aggregationbiassubmit/', aggregationbias_submit, name='aggregationbiassubmit'),


    path('fairness/1/', pp_views.lablebias_postPrediction, name='lablebiaspostprediction'),
    path('fairness/genrepo/', pp_views.generate_dissimilarity_report_postprediction, name='generate_dissimilarity_report'),
]