from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, JsonResponse
from .forms import AssociationBiasEmbeddingModel, DataSetFile
from django.views.decorators.csrf import csrf_exempt
from scipy.stats import chisquare
import pandas as pd
from fairnesstest.settings import BASE_DIR
import os
import json
import warnings
warnings.filterwarnings("ignore")
# from wefe.datasets import load_weat
from wefe.metrics import WEAT
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel
from gensim.models.keyedvectors import KeyedVectors
import gensim.downloader as api
from gensim import models
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec, FastText
from wefe.datasets import (
    load_weat,
    fetch_eds,
    fetch_debias_multiclass,
    fetch_debiaswe,
    load_bingliu,
)

dataset_file = None
@csrf_exempt
def associatebias_get_features(request):

    response = {}
    response['dataset_columns'] = []
    try:
        from ..views import common_dataset_file
        data = pd.read_csv(common_dataset_file)
        globals()['dataset_file'] = data
        response['dataset_columns'] = list(data.columns)
        response['status'] = 200
    except Exception as exp:
        response['status'] = 300
    return JsonResponse(response)

    # response = {}

    # predefined_target_set = {'Gender': ['female,woman,girl,sister,she,her,hers,daughter', 'male,man,boy,brother,he,him,his,son'],
    #                             'Sentiments':['bad,hate,wrong,negative,outcry,grudge', 'good,loved,right,positive,laughing,cheer'],
    #                             'Custom':['','']}

    # response['bias_categories'] = list(predefined_target_set.keys())

    # try:
    #     if request.method == 'GET':
    #         target = list(request.GET.keys())[0]
    #         target_set = predefined_target_set[target] 
    #         response['target_set0'] = target_set[0]
    #         response['target_set1'] = target_set[1]
    #         response['status'] = 200
    #     else:
    #         response['status'] = 200

    # except Exception as exp:
    #     response['status'] = 300

    # return JsonResponse(response)

@csrf_exempt
def save_file(file_object):
    file_name = str(file_object)
    folder_path = BASE_DIR/'structureddata//static//structureddata'
    file_path = f'{folder_path}\\files\\{file_name}'
    with open(file_path, 'wb+') as f:
        for chunk in file_object.chunks():
            f.write(chunk)
    return file_path

@csrf_exempt
def create_model(df, text_col, embedding_type, training_algo):
    df[text_col] = df[text_col].astype(str)
    corpus_text = ''.join(df[text_col])

    data = []
    # iterate through each sentence in the file
    for i in sent_tokenize(corpus_text):
        temp = []
        # tokenize the sentence into words
        for j in word_tokenize(i):
            temp.append(j.lower())
        data.append(temp)
    
    if embedding_type == 'Word2Vec':
        if training_algo == 'CBOW':
            model = Word2Vec(sg = 0)
        else:
            model = Word2Vec(sg = 1)

        model.build_vocab(data, progress_per=10000)
        model.train(data, total_examples=model.corpus_count, epochs=30, report_delay=1)
        model.init_sims(replace=True)
        model_custom = WordEmbeddingModel(model.wv, "W2V")

        return model_custom

    else:
        if training_algo == 'CBOW':
            model = FastText(sg = 0)
        else:
            model = FastText(sg = 1)

        model.build_vocab(data, progress_per=10000)
        model.train(data, total_examples=model.corpus_count, epochs=30, report_delay=1)
        model.init_sims(replace=True)
        model_custom = WordEmbeddingModel(model.wv, "W2V")

        return model_custom

WEAT_wordsets = load_weat()
RND_wordsets = fetch_eds()
# sentiments_wordsets = load_bingliu()
debias_multiclass_wordsets = fetch_debias_multiclass()

filedir = os.getcwd()
print(filedir)
filename = "unstructureddata\\associatebias"
filepath = os.path.join(filedir, filename)

# RND_wordsets = pd.read_csv(os.path.join(filepath, 'RND_wordsets.csv'))
sentiments_wordsets = pd.read_csv(os.path.join(filepath, 'sentiments_wordsets.csv'), encoding='latin-1')
# debias_multiclass_wordsets = pd.read_csv(os.path.join(filepath, 'debias_multiclass_wordsets.csv'))

gender_1 = Query(
    [RND_wordsets["male_terms"], RND_wordsets["female_terms"]],
    [WEAT_wordsets["career"], WEAT_wordsets["family"]],
    ["Male terms", "Female terms"],
    ["Career", "Family"],
)

gender_2 = Query(
    [RND_wordsets["male_terms"], RND_wordsets["female_terms"]],
    [WEAT_wordsets["math"], WEAT_wordsets["arts"]],
    ["Male terms", "Female terms"],
    ["Math", "Arts"],
)

gender_3 = Query(
    [RND_wordsets["male_terms"], RND_wordsets["female_terms"]],
    [WEAT_wordsets["science"], WEAT_wordsets["arts"]],
    ["Male terms", "Female terms"],
    ["Science", "Arts"],
)

gender_4 = Query(
    [RND_wordsets["male_terms"], RND_wordsets["female_terms"]],
    [RND_wordsets["adjectives_intelligence"], RND_wordsets["adjectives_appearance"]],
    ["Male terms", "Female terms"],
    ["Intelligence", "Appearence"],
)

gender_5 = Query(
    [RND_wordsets["male_terms"], RND_wordsets["female_terms"]],
    [RND_wordsets["adjectives_intelligence"], RND_wordsets["adjectives_sensitive"]],
    ["Male terms", "Female terms"],
    ["Intelligence", "Sensitive"],
)

gender_6 = Query(
    [RND_wordsets["male_terms"], RND_wordsets["female_terms"]],
    [WEAT_wordsets["pleasant_5"], WEAT_wordsets["unpleasant_5"]],
    ["Male terms", "Female terms"],
    ["Pleasant", "Unpleasant"],
)

gender_sent_1 = Query(
    [RND_wordsets["male_terms"], RND_wordsets["female_terms"]],
    [sentiments_wordsets["positive_words"].values, sentiments_wordsets["negative_words"].values],
    ["Male terms", "Female terms"],
    ["Positive words", "Negative words"],
)

gender_role_1 = Query(
    [RND_wordsets["male_terms"], RND_wordsets["female_terms"]],
    [
        debias_multiclass_wordsets["male_roles"],
        debias_multiclass_wordsets["female_roles"],
    ],
    ["Male terms", "Female terms"],
    ["Man Roles", "Woman Roles"],
)

gender_queries = [
    gender_1,
    gender_2,
    gender_3,
    gender_4,
    gender_5,
    gender_sent_1,
    gender_role_1,
]

rel_1 = Query(

[
    debias_multiclass_wordsets["christianity_terms"],
    debias_multiclass_wordsets["islam_terms"],
],
[WEAT_wordsets["pleasant_5"], WEAT_wordsets["unpleasant_5"]],
["Christianity terms", "Islam terms"],
["Pleasant", "Unpleasant"],
)

rel_2 = Query(
    [
        debias_multiclass_wordsets["christianity_terms"],
        debias_multiclass_wordsets["judaism_terms"],
    ],
    [WEAT_wordsets["pleasant_5"], WEAT_wordsets["unpleasant_5"]],
    ["Christianity terms", "Judaism terms"],
    ["Pleasant", "Unpleasant"],
)

rel_3 = Query(
    [
        debias_multiclass_wordsets["islam_terms"],
        debias_multiclass_wordsets["judaism_terms"],
    ],
    [WEAT_wordsets["pleasant_5"], WEAT_wordsets["unpleasant_5"]],
    ["Islam terms", "Judaism terms"],
    ["Pleasant", "Unpleasant"],
)

rel_4 = Query(
    [
        debias_multiclass_wordsets["christianity_terms"],
        debias_multiclass_wordsets["islam_terms"],
    ],
    [
        debias_multiclass_wordsets["conservative"],
        debias_multiclass_wordsets["terrorism"],
    ],
    ["Christianity terms", "Islam terms"],
    ["Conservative", "Terrorism"],
)

rel_5 = Query(
    [
        debias_multiclass_wordsets["christianity_terms"],
        debias_multiclass_wordsets["judaism_terms"],
    ],
    [debias_multiclass_wordsets["conservative"], debias_multiclass_wordsets["greed"]],
    ["Christianity terms", "Jew terms"],
    ["Conservative", "Greed"],
)

rel_6 = Query(
    [
        debias_multiclass_wordsets["islam_terms"],
        debias_multiclass_wordsets["judaism_terms"],
    ],
    [debias_multiclass_wordsets["terrorism"], debias_multiclass_wordsets["greed"]],
    ["Islam terms", "Jew terms"],
    ["Terrorism", "Greed"],
)

rel_sent_1 = Query(
    [
        debias_multiclass_wordsets["christianity_terms"],
        debias_multiclass_wordsets["islam_terms"],
    ],
    [sentiments_wordsets["positive_words"].values, sentiments_wordsets["negative_words"].values],
    ["Christianity terms", "Islam terms"],
    ["Positive words", "Negative words"],
)

rel_sent_2 = Query(
    [
        debias_multiclass_wordsets["christianity_terms"],
        debias_multiclass_wordsets["judaism_terms"],
    ],
    [sentiments_wordsets["positive_words"].values, sentiments_wordsets["negative_words"].values],
    ["Christianity terms", "Jew terms"],
    ["Positive words", "Negative words"],
)

rel_sent_3 = Query(
    [
        debias_multiclass_wordsets["islam_terms"],
        debias_multiclass_wordsets["judaism_terms"],
    ],
    [sentiments_wordsets["positive_words"].values, sentiments_wordsets["negative_words"].values],
    ["Islam terms", "Jew terms"],
    ["Positive words", "Negative words"],
)

religion_queries = [
    rel_1,
    rel_2,
    rel_3,
    rel_4,
    rel_5,
    rel_6,
    rel_sent_1,
    rel_sent_2,
    rel_sent_3,
]

eth_1 = Query(
    [RND_wordsets["names_white"], RND_wordsets["names_black"]],
    [WEAT_wordsets["pleasant_5"], WEAT_wordsets["unpleasant_5"]],
    ["White last names", "Black last names"],
    ["Pleasant", "Unpleasant"],
)

eth_2 = Query(
    [RND_wordsets["names_white"], RND_wordsets["names_asian"]],
    [WEAT_wordsets["pleasant_5"], WEAT_wordsets["unpleasant_5"]],
    ["White last names", "Asian last names"],
    ["Pleasant", "Unpleasant"],
)

eth_3 = Query(
    [RND_wordsets["names_white"], RND_wordsets["names_hispanic"]],
    [WEAT_wordsets["pleasant_5"], WEAT_wordsets["unpleasant_5"]],
    ["White last names", "Hispanic last names"],
    ["Pleasant", "Unpleasant"],
)

eth_4 = Query(
    [RND_wordsets["names_white"], RND_wordsets["names_black"]],
    [RND_wordsets["occupations_white"], RND_wordsets["occupations_black"]],
    ["White last names", "Black last names"],
    ["Occupations white", "Occupations black"],
)

eth_5 = Query(
    [RND_wordsets["names_white"], RND_wordsets["names_asian"]],
    [RND_wordsets["occupations_white"], RND_wordsets["occupations_asian"]],
    ["White last names", "Asian last names"],
    ["Occupations white", "Occupations asian"],
)

eth_6 = Query(
    [RND_wordsets["names_white"], RND_wordsets["names_hispanic"]],
    [RND_wordsets["occupations_white"], RND_wordsets["occupations_hispanic"]],
    ["White last names", "Hispanic last names"],
    ["Occupations white", "Occupations hispanic"],
)

eth_sent_1 = Query(
    [RND_wordsets["names_white"], RND_wordsets["names_black"]],
    [sentiments_wordsets["positive_words"].values, sentiments_wordsets["negative_words"].values],
    ["White last names", "Black last names"],
    ["Positive words", "Negative words"],
)

eth_sent_2 = Query(
    [RND_wordsets["names_white"], RND_wordsets["names_asian"]],
    [sentiments_wordsets["positive_words"].values, sentiments_wordsets["negative_words"].values],
    ["White last names", "Asian last names"],
    ["Positive words", "Negative words"],
)

eth_sent_3 = Query(
    [RND_wordsets["names_white"], RND_wordsets["names_hispanic"]],
    [sentiments_wordsets["positive_words"].values, sentiments_wordsets["negative_words"].values],
    ["White last names", "Hispanic last names"],
    ["Positive words", "Negative words"],
)

ethnicity_queries = [
    eth_1,
    eth_2,
    eth_3,
    eth_4,
    eth_5,
    eth_6,
    eth_sent_1,
    eth_sent_2,
    eth_sent_3,
]


@csrf_exempt
def associatebias_submit(request):
    pass
    '''
    This will generate the association bias output on user inputs
    '''
    try:
        response = {}
        if request.method == 'POST':

            text_col = request.POST['text_col']
            embedding_type = request.POST['embedding_type']
            training_algo = request.POST['training_algo']

            print(text_col, embedding_type, training_algo)

            print('Reading Word Embeddings File for Association Bias Test')
            embeddings_file = dataset_file
            
            print(f'Processing {embeddings_file} file')

            try:
                model = create_model(embeddings_file[[text_col]], text_col, embedding_type, training_algo)
            except:
                response['status'] = 400
                return JsonResponse(response)

            print('model is here')

            temp_weat = []
            gender_output = []
            ethnicity_output = []
            religion_output = []

            print('Running Gender Queries..')
            for j in gender_queries:
                weat = WEAT()
                result = weat.run_query(j, model)

                txt = result["query_name"]
                t1 = txt.split('wrt')[0].split('and')[0].strip()
                t2 = txt.split('wrt')[0].split('and')[1].strip()
                a1 = txt.split('wrt')[1].split('and')[0].strip()
                a2 = txt.split('wrt')[1].split('and')[1].strip()

                if str(abs(result['weat'])) != 'nan':
                    temp_weat.append(abs(result['weat']))
                    if result['weat']>= 0.1:
                        # gender_output.append(f'There is an association detected between {t1} with {a1} and {t2} with {a2}.')
                        gender_output.append([t1, a1, "missing"])
                        gender_output.append([t2, a2, "missing"])
                    elif result['weat'] <= -0.1:
                        # gender_output.append(f'There is an association detected between {t1} with {a2} and {t2} with {a1}.')
                        gender_output.append([t1, "missing", a2])
                        gender_output.append([t2, "missing", a1])
                    else:
                        pass
                
            df_gender_output = pd.DataFrame(gender_output, columns = ['Gender', 'Positive Association', 'Negative Association'])
            df_gender_output = df_gender_output.drop_duplicates()

            # concatenate the string for each gender
            df_gender_output['Positive Association'] = df_gender_output.groupby(['Gender'])['Positive Association'].transform(lambda x: ', '.join([i for i in x if i != "missing"]))
            df_gender_output['Negative Association'] = df_gender_output.groupby(['Gender'])['Negative Association'].transform(lambda x: ', '.join([i for i in x if i != "missing"]))
                        
            df_gender_output = df_gender_output.drop_duplicates()

            result_gender = '''<style>
                            .df th { background-color: #a100ff; color: white;}
                            table {
                            width: 50%;
                            }
                            </style>''' + df_gender_output.to_html(border = 2, justify = 'left', classes=['table table-stripped', 'df'], index = False)
            
            print(df_gender_output)

            print('Running Religion Queries..')
            for j in religion_queries:
                weat = WEAT()
                result = weat.run_query(j, model)

                txt = result["query_name"]
                t1 = txt.split('wrt')[0].split('and')[0].strip()
                t2 = txt.split('wrt')[0].split('and')[1].strip()
                a1 = txt.split('wrt')[1].split('and')[0].strip()
                a2 = txt.split('wrt')[1].split('and')[1].strip()

                if str(abs(result['weat'])) != 'nan':
                    temp_weat.append(abs(result['weat']))
                    if result['weat']>= 0.1:
                        # religion_output.append(f'There is an association detected between {t1} with {a1} and {t2} with {a2}.')
                        religion_output.append([t1, a1, "missing"])
                        religion_output.append([t2, a2, "missing"])
                    elif result['weat'] <= -0.1:
                        # religion_output.append(f'There is an association detected between {t1} with {a2} and {t2} with {a1}.')
                        religion_output.append([t1, "missing", a2])
                        religion_output.append([t2, "missing", a1])
                    else:
                        pass
            
            df_religion_output = pd.DataFrame(religion_output, columns = ['Religion', 'Positive Association', 'Negative Association'])
            df_religion_output = df_religion_output.drop_duplicates()

            # concatenate the string for each religion
            df_religion_output['Positive Association'] = df_religion_output.groupby(['Religion'])['Positive Association'].transform(lambda x: ', '.join([i for i in x if i != "missing"]))
            df_religion_output['Negative Association'] = df_religion_output.groupby(['Religion'])['Negative Association'].transform(lambda x: ', '.join([i for i in x if i != "missing"]))
                        
            df_religion_output = df_religion_output.drop_duplicates()

            result_religion = '''<style>
                            .df th { background-color: #a100ff; color: white;}
                            table {
                            width: 50%;
                            }
                            </style>''' + df_religion_output.to_html(border = 2, justify = 'left', classes=['table table-stripped','df'], index = False)

            print(df_religion_output)

            print('Running Ethnicity Queries..')
            for j in ethnicity_queries:
                weat = WEAT()
                result = weat.run_query(j, model)
                
                txt = result["query_name"]
                t1 = txt.split('wrt')[0].split('and')[0].strip()
                t2 = txt.split('wrt')[0].split('and')[1].strip()
                a1 = txt.split('wrt')[1].split('and')[0].strip()
                a2 = txt.split('wrt')[1].split('and')[1].strip()

                if str(abs(result['weat'])) != 'nan':
                    temp_weat.append(abs(result['weat']))
                    if result['weat']>= 0.1:
                        # ethnicity_output.append(f'There is an association detected between {t1} with {a1} and {t2} with {a2}.')
                        ethnicity_output.append([t1, a1, "missing"])
                        ethnicity_output.append([t2, a2, "missing"])
                    elif result['weat'] <= -0.1:
                        # ethnicity_output.append(f'There is an association detected between {t1} with {a2} and {t2} with {a1}.')
                        ethnicity_output.append([t1, "missing", a2])
                        ethnicity_output.append([t2, "missing", a1])
                    else:
                        pass 

            df_ethnicity_output = pd.DataFrame(ethnicity_output, columns = ['Ethnicity', 'Positive Association', 'Negative Association'])
            df_ethnicity_output = df_ethnicity_output.drop_duplicates()

            # concatenate the string for each ethnicity
            df_ethnicity_output['Positive Association'] = df_ethnicity_output.groupby(['Ethnicity'])['Positive Association'].transform(lambda x: ', '.join([i for i in x if i != "missing"]))
            df_ethnicity_output['Negative Association'] = df_ethnicity_output.groupby(['Ethnicity'])['Negative Association'].transform(lambda x: ', '.join([i for i in x if i != "missing"]))
                        
            df_ethnicity_output = df_ethnicity_output.drop_duplicates()

            result_ethnicity = '''<style>
                            .df th { background-color: #a100ff; color: white;}
                            table {
                            width: 50%;
                            }
                            </style>''' + df_ethnicity_output.to_html(border = 2, justify = 'left', classes=['table table-stripped','df'], index = False)

            print(df_ethnicity_output)

            response['ethnicity_output'] = result_ethnicity #"<br>".join(ethnicity_output)
            response['gender_output'] = result_gender #"<br>".join(gender_output)
            response['religion_output'] = result_religion #"<br>".join(religion_output)
            response['status'] = 200

            if len(ethnicity_output) == 0:
                response['ethnicity_output'] = 'No Association Bias Detected.'
        
            if len(gender_output) == 0:
                response['gender_output'] = 'No Association Bias Detected.'

            if len(religion_output) == 0:
                response['religion_output'] = 'No Association Bias Detected.'

            print('Association Bias Test successfully ran.')
            return JsonResponse(response)

    except:
        response['status'] = 300
        return JsonResponse(response)