#import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest as ztest
from scipy.stats import chi2_contingency, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import math
import mpld3
sns.set_style('darkgrid')

def test_independence(data, protectedVar, targetVar):
    '''
    Returns the p-value corresponding to test of independence between a protected variable
    and the target variable.

    Parameters:
    - data [pd.DataFrame] = data under investigation
    - protectedVar [str] = column name for protected attribute
    - targetVar [str] = column name for target variable
    '''

    if len(data[protectedVar].unique()) == 2:
        vala, valb = data[protectedVar].unique()
        count = (data.loc[(data[targetVar] == 1) & (data[protectedVar] == vala), :].shape[0],
                 data.loc[(data[targetVar] == 1) & (data[protectedVar] == valb), :].shape[0])
        nobs = (data.loc[data[protectedVar] == vala, :].shape[0], data.loc[data[protectedVar] == valb, :].shape[0])
        _, p = ztest(count, nobs, alternative = 'two-sided')
    elif len(data[protectedVar].unique()) < 10:
        p = chi2_contingency(pd.crosstab(data[targetVar], data[protectedVar]))[1]
    elif len(data[protectedVar].unique()) > 10:
        p = ttest_ind(data.loc[data[targetVar] == 0, protectedVar].values,
                      data.loc[data[targetVar] == 1, protectedVar].values)[1]
    return p


def plotProbDist(protectedVar_categorical, protectedVar_continuous, targetVar, data, confidence_interval:float = 0.5):
    '''
    Generates distributions of protected variable categories across target variable classes.

    Parameters:
    - protectedVar_categorical: List of str indicating protected categorical attributes
    - protectedVar_continuous: List of str indicating protected continuous attributes
    - targetVar: str indicating column name for target variable
    - data: pandas.DataFrame under investigation
    - Confidence Interval as selected by the user

    Returns:
    - None

    '''
    #print(type(data))
    #data.to_csv('./pre_pred_labelbias.csv', index=False)
    total_num_plots = len(protectedVar_categorical) + len(protectedVar_continuous)
    c = sns.color_palette('Set2')
    # statements, pvalues = [], {} #pvalues dictionary contains feature and it's pvalue
    statements, pvalues, label_bias_graphs = [], {}, [] #pvalues dictionary contains feature and it's pvalue
    for j, col in enumerate(protectedVar_categorical + protectedVar_continuous):
        
        pvalue = test_independence(data, col, targetVar)
        if pvalue >= confidence_interval:
            statements.append(f'p-value for independence (null hypothesis) on {targetVar} for {col} = {round(pvalue,2)}')
            pvalues[col] = pvalue
        
        
        if col in protectedVar_categorical:
            crosstab = pd.crosstab(index = data[col], columns = data[targetVar])
            normed_crosstab = pd.crosstab(index = data[col], columns = data[targetVar], normalize = 'index')

            fig, ax = plt.subplots()
            fig.set_figheight(4)
            fig.set_figwidth(15)
            normed_crosstab.plot(kind = 'bar', stacked = True, ax = ax, color = c)
            ax.set_ylabel('percentage');
            ax.set_title(f'Distribution of {col} with respect to {targetVar}');

            prev_heights = []
            for i, bar in enumerate(ax.get_children()[:2]):
                ax.annotate(f'{crosstab.values[i, 0]} ({round(bar.get_height() * 100, 3)} %)',
                           (bar.get_x() + bar.get_width() / 2, bar.get_height() / 2),
                            ha = 'center', va = 'bottom');
                prev_heights.append(bar.get_height())

            for i, bar in enumerate(ax.get_children()[2:4]):
                ax.annotate(f'{crosstab.values[i-2, 1]} ({round(bar.get_height() * 100, 3)} %)',
                           (bar.get_x() + bar.get_width() / 2, prev_heights[i] + 0.0125),
                            ha = 'center', va = 'bottom');
            

        else:
            fig, ax = plt.subplots()
            fig.set_figheight(6)
            fig.set_figwidth(15)
            sns.violinplot(x = targetVar, y = col, data = data, ax = ax, palette = c);
            ax.set_title(f'Distribution of {col} with respect to {targetVar}');
            
        # plt.savefig(f'structureddata\\static\\structureddata\\label_bias_images\\label_bias_{j+1}.png')
        label_bias_graphs.append(mpld3.fig_to_html(fig))

    statements.append(f'For rest of the features, p-value is < {confidence_interval}.')
    
    #View pvalue graph
    # _ = pscore_graph(pvalues, targetVar, total_num_plots) #Replace 0.5 with user selected confidence interval
        
    # return statements
    pre_pred_label_bias = pscore_graph(pvalues, targetVar, total_num_plots)
    return statements, label_bias_graphs, pre_pred_label_bias

def pscore_graph(feature_pscore_vals:dict, target_var:str, total_num_feat:int):
    '''This function takes dictionary, with key as feature name and value as pvalues, and creates bar chart.
    '''
    
    #Sorting features based on scores
    feature_pscore_vals = dict(sorted(list(feature_pscore_vals.items()), key = lambda val: val[1]))

    grid_dict = {'width_ratios': [0.75, 0.25], 'wspace': 0.1, 'hspace': 0.1}
    fig, (ax, ax2) = plt.subplots(1, 2, constrained_layout=True, gridspec_kw = grid_dict)

    ax.set_facecolor('black')

    #Make figure width dynamic based on number of parameters
    num_feat = len(feature_pscore_vals)
    min_graph_bars = 7
    if num_feat < min_graph_bars:
        lower_fake_feat_num = math.floor((min_graph_bars - num_feat)/2)
        upper_fake_feat_num = math.ceil((min_graph_bars - num_feat)/2)
        fig_height = 0.6*min_graph_bars
    else:
        lower_fake_feat_num, upper_fake_feat_num = 0, 0
        fig_height = 0.6*num_feat
  
    fig.set_figheight(fig_height)
    #fig.patch.set_facecolor('#f3f3f3')
    fig.patch.set_facecolor('black')

    px = 1/plt.rcParams['figure.dpi']  # pixel in inches

    fig.set_figwidth(1800*px) #Setting figure width based in window width in pixels. 1080 should come from frontend

    x_labels = ['']*lower_fake_feat_num + \
                list(feature_pscore_vals.keys()) + \
                ['']*upper_fake_feat_num

    ax.grid(visible=True, axis='x', zorder=0, ls='--', fillstyle='bottom')
    ax.grid(visible=False, axis='y', zorder=0, ls='-', fillstyle='bottom')

    ax.barh(y = [round(i/10 + 0.1, 1) for i in range(0, num_feat+lower_fake_feat_num+upper_fake_feat_num)], 
            width = [0]*lower_fake_feat_num + list(feature_pscore_vals.values()) + [0]*upper_fake_feat_num, 
            height=0.07, color='darkviolet', tick_label = [' ']*(num_feat+lower_fake_feat_num+upper_fake_feat_num), 
            rasterized=False, edgecolor = '#f0e6f2', zorder=2, alpha = 0.8
            )

    _ = ax.set_title('P-Scores', fontweight ='bold', fontfamily='Graphik', fontsize = 16, color='white')

    ax.set_xlim(0,1)
    ax.set_xticks([0.1, 0.2, 0.3, 0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    ax.tick_params(left = False, axis='x', colors='white', labelsize=12)
    ax.spines['right'].set_color('white')
    ax.xaxis.label.set_color('white')

    for bar, feature_name in zip(ax.patches, x_labels):
        ax.text(0.02, bar.get_y()+bar.get_height()/2, feature_name, color = 'white', ha = 'left', va = 'center', 
                fontfamily='Graphik', fontsize=12, fontweight='bold')
        
    ##Making donut graph
    # colors
    colors = ['grey','darkviolet']

    # explosion
    explode = (0.03, 0.03)

    # Pie Chart
    num_gt_threshold = len(feature_pscore_vals)
    num_lt_threshold = total_num_feat - num_gt_threshold
    ax2.pie([num_gt_threshold, num_lt_threshold], colors=colors, labels=['Possible Biased', 'Possibly Unbiased'],
            autopct='%1.1f%%', pctdistance=0.7, explode=explode, startangle=90, radius = 1.2,
            wedgeprops={'edgecolor':'#f0e6f2'},
            textprops={'color':'white', 'fontweight':'bold', 'fontfamily':'Graphik', 'fontsize':14}
            )

    _ = ax2.set_title('(Un)Biased Features Count (%)', fontweight ='bold', fontfamily='Graphik', fontsize = 16, 
                    color = 'white', pad=25
                    )

    # draw circle
    centre_circle = plt.Circle((0, 0), 0.55, fc='black', ec='#f0e6f2')
    
    # Adding Circle in Pie chart
    _=fig.gca().add_artist(centre_circle)
    
    # plt.savefig(f'structureddata\\static\\structureddata\\label_bias_images\\pre_pred_label_bias.png')

    # return None
    return mpld3.fig_to_html(fig)