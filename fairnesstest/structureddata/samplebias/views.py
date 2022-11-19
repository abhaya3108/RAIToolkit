import json
from turtle import color
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse, JsonResponse
from scipy.stats import chisquare
from fairnesstest.settings import BASE_DIR
from ..misc.preprocessing import preprocess_data

from matplotlib import pyplot as plt
import mpld3
import pandas as pd
import math
import pickle

dataset_file = None

@csrf_exempt
def samplebias_getfeatures(request):
    response = {}
    response['dataset_columns'] = []
    try:
        from ..views import common_dataset_file, common_dataframe
        # data = pd.read_csv(common_dataset_file)
        globals()['dataset_file'] = common_dataframe #preprocess_data(data)
        response['dataset_columns'] = list(dataset_file.columns)
        response['status'] = 200
    except Exception as e:
        response['error'] = str(e)
        response['status'] = 300
    return JsonResponse(response)

@csrf_exempt
def samplebias_submit(request):
    '''
    This will generate the sample bias output on user selections
    '''
    if request.method == 'POST':
        response = {}
        samplebias_output = []
        requestdata = json.loads(request.body)
        #print("request.body :", requestdata)
        selected_features = requestdata['selected_features']
        width = requestdata['width']
        '''
        data_prep = preprocess_data(data = dataset_file[selected_features])
        # data_prep = dataset_file
        features_of_interest = data_prep.columns
        score_threshold = 0.4 #Get it from user
        
        feature_probability_vals = {} #dict with feature name and probability value
        samplebias_output = [] #List contains output statements to be shown to user
        for col in features_of_interest:
            score = test_sample_bias(data_prep, col)
            if score > score_threshold:
                feature_probability_vals[col] = round(score, 2)
            else:
                pass
        
        if len(feature_probability_vals) == 0:
            samplebias_output.append(f'For all selected features, chances of being biased is less than {score_threshold}.')
        else:
            samplebias_output = [f'For {col}: {score}' for col, score in feature_probability_vals.items()]
            samplebias_output.append(f'For rest of selected features: < {score_threshold}')
        
        print('Output Statement: \n', samplebias_output)
        
        if len(feature_probability_vals) >= 2:
            count_feat_gte_threshold = len(feature_probability_vals)
            count_feat_lt_threshold = len(features_of_interest) - count_feat_gte_threshold
            # html_fig_str = sample_bias_graph(feature_probability_vals, count_feat_gte_threshold, count_feat_lt_threshold)
            html_fig_str = sample_bias_graph(feature_probability_vals, count_feat_gte_threshold, count_feat_lt_threshold, width)
            sample_graph = 1 #True
        else:
            html_fig_str = None
            sample_graph = 0 #False
        
        '''
        data_prep = preprocess_data(data = dataset_file[selected_features])
        samplebias_output, html_fig_str, sample_graph = samplebias_calculations(data_prep, selected_features, width)

        response['samplebias_output'] = samplebias_output
        response['html_samplebias_fig'] = html_fig_str #Temporary for testing
        response['is_show_samplebias_image'] = sample_graph
        response['selected_features'] = ", ".join(selected_features)
        response['samplebias_graph'] = sample_graph
        response['status'] = 200

    else:
        response['samplebias_output'] = []
        response['html_samplebias_fig'] = '' #Temporary for testing
        response['is_show_samplebias_image'] = 0
        response['selected_features'] = ", ".join(selected_features)
        response['samplebias_graph'] = 0
        response['status'] = 500
        
    return JsonResponse(response)

def samplebias_calculations(data_prep, selected_features, width, score_threshold = 0.1):
    
    # data_prep = dataset_file
    features_of_interest = data_prep.columns
    #score_threshold = 0.4 #Get it from user
    
    feature_probability_vals = {} #dict with feature name and probability value
    samplebias_output = [] #List contains output statements to be shown to user
    for col in features_of_interest:
        score = test_sample_bias(data_prep, col)
        print(f'{col}: {score}')
        if score > score_threshold:
            feature_probability_vals[col] = round(score, 2)
        else:
            pass
    
    if len(feature_probability_vals) == 0:
        samplebias_output.append(f'For all selected features, chances of being biased is less than {score_threshold}.')
    else:
        samplebias_output = [f'For {col}: {score}' for col, score in feature_probability_vals.items()]
        samplebias_output.append(f'For rest of selected features: < {score_threshold}')
    
    print('Output Statement: \n', samplebias_output)
    
    if len(feature_probability_vals) >= 2:
        count_feat_gte_threshold = len(feature_probability_vals)
        count_feat_lt_threshold = len(features_of_interest) - count_feat_gte_threshold
        # html_fig_str = sample_bias_graph(feature_probability_vals, count_feat_gte_threshold, count_feat_lt_threshold)
        html_fig_str = sample_bias_graph(feature_probability_vals, count_feat_gte_threshold, count_feat_lt_threshold, width)
        sample_graph = 1 #True
    else:
        html_fig_str = None
        sample_graph = 0 #False

    return samplebias_output, html_fig_str, sample_graph

def test_sample_bias(data, column):
            '''
            Column must be categorical. If numerical column, bin the values first else this will not work.
            '''
            return 1 - chisquare(f_obs = data[column])[1]

# def sample_bias_graph(feature_probability_vals:dict, num_feat_gte_thresh:int, num_feat_lt_thresh:int):
def sample_bias_graph(feature_probability_vals:dict, num_feat_gte_thresh:int, num_feat_lt_thresh:int, width=None):
    '''This function generates takes dictionary, with key as feature name and value as probability, and creates bar chart.
    '''
    #Sorting features by values
    feature_probability_vals = dict(sorted(list(feature_probability_vals.items()), key = lambda val: val[1]))
    
    grid_dict = {'width_ratios': [0.75, 0.25], 'wspace': 0.1, 'hspace': 0.1}
    fig, (ax, ax2) = plt.subplots(1, 2, constrained_layout=True, gridspec_kw = grid_dict)

    # ax.set_facecolor('black')

    #Make figure width dynamic based on number of parameters

    num_feat = len(feature_probability_vals)
    min_graph_bars = 7
    if num_feat < min_graph_bars:
        lower_fake_feat_num = math.floor((min_graph_bars - num_feat)/2)
        upper_fake_feat_num = math.ceil((min_graph_bars - num_feat)/2)
        fig_height = 0.6*min_graph_bars
    else:
        lower_fake_feat_num, upper_fake_feat_num = 0, 0
        fig_height = 0.6*num_feat
    
    # fig.set_figheight(fig_height)
    #fig.patch.set_facecolor('#f3f3f3')
    # fig.patch.set_facecolor('black')

    px = 1/plt.rcParams['figure.dpi']  # pixel in inches

    # fig.set_figwidth(1200*px) #Setting figure width based in window width in pixels. 1080 should come from frontend
    fig.set_figwidth(0.9*width*px)

    x_labels = (['']*lower_fake_feat_num) + \
                [f'{str(feat_name)}' + ': ' + f'{str(val)}' for feat_name, val in feature_probability_vals.items()] + \
                (['']*upper_fake_feat_num)

    ax.grid(visible=True, axis='x', zorder=0, ls='-', color='#f3f3f3')
    ax.grid(visible=False, axis='y')
    ax.barh(y = [round(i/10 + 0.1, 1) for i in range(0, num_feat+lower_fake_feat_num+upper_fake_feat_num)], 
            width=([0]*lower_fake_feat_num) + list(feature_probability_vals.values()) + ([0]*upper_fake_feat_num), 
            height=0.07, color='darkviolet', tick_label = [' ']*(num_feat+lower_fake_feat_num+upper_fake_feat_num), 
            rasterized=True, edgecolor = '#f0e6f2', 
            zorder=2, alpha = 0.8
            )

    _ = ax.set_title('Confidence Scores', fontweight ='bold', fontfamily='Graphik', fontsize = 16)
    
    ax.set_xlim(0,1)
    ax.set_xticks([0.1, 0.2, 0.3, 0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    ax.tick_params(left = False, axis='x', labelsize=12)
    # ax.spines['right'].set_color('white')
    # ax.xaxis.label.set_color('white')
    ax.set_yticklabels([])

    for bar, feature_name in zip(ax.patches, x_labels):
        ax.text(0.02, bar.get_y()+bar.get_height()/2, feature_name, ha = 'left', va = 'center', 
                fontfamily='Graphik', fontsize=12, fontweight='bold')

    ##Making donut graph
    colors = ['grey','darkviolet']
    
    explode = (0.03, 0.03)

    # Pie Chart
    ax2.pie([num_feat_gte_thresh, num_feat_lt_thresh], colors=colors, labels=['Possibly Biased', 'Possibly Unbiased'], 
            autopct=lambda p: '{:.1f}% ({:.0f})'.format(p,(p*(num_feat_gte_thresh+num_feat_lt_thresh))/100), pctdistance=0.7, 
            explode=explode, startangle=90, radius = 1.2,
            wedgeprops={'edgecolor':'#f0e6f2'}, 
            textprops={'fontweight':'bold', 'fontfamily':'Graphik', 'fontsize':14}
            )

    _ = ax2.set_title('(Un)Biased Features Count (%)', fontweight ='bold', fontfamily='Graphik', 
                    fontsize = 16, pad=20
                    )
    
    # draw circle
    centre_circle = plt.Circle((0, 0), 0.55, fc='white', ec='#f0e6f2')
    
    # Adding Circle in Pie chart
    _ = fig.gca().add_artist(centre_circle)
    
    # fig.savefig(BASE_DIR/'structureddata/static/structureddata/images/sample_bias_images/sample_bias.png', transparent=False)

    return mpld3.fig_to_html(fig)
