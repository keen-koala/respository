# -*- coding: utf-8 -*-

from csv import reader, writer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import csv
import re
from removeaccents import removeaccents
import unicodedata
import os

filename = "C:/Users/Vanessa/Desktop/Masterarbeit/MA_repository/respository/idiom_sentences_multilingual_test.tsv"
delete_character_lits=["</s><s>", "'", ]
sentiment_annotator_list= ["Cardiff_NLP", "NLP_town", "Vader"]
Cardiff_NLP_list=[]
NLP_town_list=[]
Vader_list=[]
testlist=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,1]

''' With pandas'''



#csv_input.insert(2, 'test', [test for test in testlist])
#print(csv_input.head())
#sentence= csv_input['Left']+ csv_input['KWIC']+ csv_input['Right']


#csv_input['New Value'] = [test for test in testlist]
#csv_input.to_csv('testcase3.csv', index=False)

csv_input = pd.read_csv(filename,sep="\t",header=0)
for index, row in csv_input.iterrows():
    index = row[0]
    sentence= (row["Left"], row["KWIC"], row['Right'])
    sentence=str(sentence)
    #print(index)
    #print(sentence)

    char_to_replace = {'</s><s>': '',
                             "''": "'"
                             }#https://thispointer.com/python-replace-multiple-characters-in-a-string/
    # Iterate over all key-value pairs in dictionary
    for key, value in char_to_replace.items():
        #     # Replace key character with value character in string
        sentence = sentence.replace(key, value)
        sentence_lower = str(sentence).lower()  # convert to lowercase for processing in sentiment annotator
        sentence_no_acccents = removeaccents.remove_accents(sentence_lower)
    print(index, ':', sentence_no_acccents)

    ''' Pipeline with CardiffNLP'''
    model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    sentiment_task_cardiff = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
    cardiff_sentiment_label = (sentiment_task_cardiff(sentence_no_acccents)[0]['label'])
    print('Cardiff NLP: ', str(cardiff_sentiment_label))
    if cardiff_sentiment_label== 'Neutral':
        cardiff_sentiment_label== '0.0'
        Cardiff_NLP_list.append(cardiff_sentiment_label)
    elif cardiff_sentiment_label== 'Negative':
        cardiff_sentiment_label == '-1.0'
        Cardiff_NLP_list.append(cardiff_sentiment_label)
    else:
        cardiff_sentiment_label == '-1.0'
        Cardiff_NLP_list.append(cardiff_sentiment_label)


    ''' NLP town Model '''  # Instantiate a pipeline object with our task and model passed as parameters,  #https://www.kdnuggets.com/2021/06/create-deploy-sentiment-analysis-app-api.html

    sentiment_task_nlptown = pipeline(task='sentiment-analysis',
                                      model='nlptown/bert-base-multilingual-uncased-sentiment')
    nlp_town_sentiment_label = sentiment_task_nlptown(sentence_no_acccents)[0]['label']
    NLP_town_list.append(nlp_town_sentiment_label)

    # Pass the text to our pipeline and print the results
    print('NLP town', f'{nlp_town_sentiment_label}')

    ''' Using Vader'''  # https://github.com/cjhutto/vaderSentiment
    # using vader multihttps://pypi.org/project/vader-multi/
    # https://github.com/brunneis/vader-multi

    analyzer = SentimentIntensityAnalyzer()
    vader_sentiment_label = analyzer.polarity_scores(sentence_no_acccents)['compound']
    Vader_list.append(vader_sentiment_label)
    print('Vader: ', vader_sentiment_label)

#print(NLP_town_list)
#print(Cardiff_NLP_list)
#print(Vader_list)


csv_input['NLP_town'] = [label for label in NLP_town_list]
csv_input['Cardiff_NLP'] = [label for label in Cardiff_NLP_list]
csv_input['Vader'] = [label for label in Vader_list]
csv_input.to_csv('sentiment_test_short_labels.csv', index=False)








'''Old pipeline '''


# with open(filename, 'r', encoding='utf-8') as tsvfile:  # https://stackoverflow.com/questions/44251813/unicodedecodeerror-charmap-codec-cant-decode-byte-0x9d-in-position-1010494
#     datareader = csv.reader(tsvfile, delimiter="\t")
#     next(datareader,None)  # https://stackoverflow.com/questions/14257373/skip-the-headers-when-editing-a-csv-file-using-python
#     #tsvfile.seek(0)  # back to the start of the CSV file # https://stackoverflow.com/questions/31537516/loop-through-tsv-csv-file-stops-after-run
#
#
#     for columns in datareader:
#         #next(tsvfile)
#         column1, column2, column3, column4, column5, column6 = columns
#         sentence = ((column4), (column5), (column6))
#         sentence = str(sentence) # convert tuple to string
#         #print('sentence original: ', sentence)
#         index = column1
#         #replace characters that are not wanted
#         char_to_replace = {'</s><s>': '',
#                            "''": "'"
#                            }#https://thispointer.com/python-replace-multiple-characters-in-a-string/
#         # Iterate over all key-value pairs in dictionary
#         for key, value in char_to_replace.items():
#             # Replace key character with value character in string
#             sentence = sentence.replace(key, value)
#
#         sentence_lower = str(sentence).lower()  # convert to lowercase for processing in sentiment annotator
#         #print('sentence replaces: ', sentence_lower)
#
#    #remove accents
#         # s_no_accents = ''.join((c for c in unicodedata.normalize('NFD', sentence_lower) if unicodedata.category(c) != 'Mn'))
#         # print(s_no_accents)
#         sentence_no_acccents = removeaccents.remove_accents(sentence_lower)
#         print( index, ':', sentence_no_acccents)
#
#
#
#
#         ''' Pipeline with CardiffNLP'''
#         model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
#         sentiment_task_cardiff = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
#         cardiff_sentiment_label = (sentiment_task_cardiff(sentence_no_acccents))
#         print('Cardiff NLP: ', str(sentiment_task_cardiff(sentence_no_acccents)))
#         Cardiff_NLP_list.append(cardiff_sentiment_label)
#
#
#         ''' NLP town Model '''  # Instantiate a pipeline object with our task and model passed as parameters,  #https://www.kdnuggets.com/2021/06/create-deploy-sentiment-analysis-app-api.html
#
#         sentiment_task_nlptown = pipeline(task='sentiment-analysis',
#                                           model='nlptown/bert-base-multilingual-uncased-sentiment')
#         nlp_town_sentiment_label= sentiment_task_nlptown(sentence_no_acccents)
#         NLP_town_list.append(nlp_town_sentiment_label)
#
#         # Pass the text to our pipeline and print the results
#         print('NLP town', f'{sentiment_task_nlptown(sentence_no_acccents)}')
#
#         ''' Using Vader'''  # https://github.com/cjhutto/vaderSentiment
#         # using vader multihttps://pypi.org/project/vader-multi/
#         # https://github.com/brunneis/vader-multi
#
#         analyzer = SentimentIntensityAnalyzer()
#         vader_sentiment_label = analyzer.polarity_scores(sentence_no_acccents)
#         Vader_list.append(vader_sentiment_label)
#         print('Vader: ', vader_sentiment_label)
#
# print(NLP_town_list)
# print(Cardiff_NLP_list)
# print(Vader_list)












# vader_sentiment = []
# cardiff_sentiment=[]
# nlp_town_sentiment=[]
#
#
# #https://thispointer.com/python-add-a-column-to-an-existing-csv-file/
# def add_column_in_csv(input_file, output_file, transform_row):
#     """ Append a column in existing csv using csv.reader / csv.writer classes"""
#     # Open the input_file in read mode and output_file in write mode
#     with open(input_file, 'r',encoding='latin-1') as read_obj, \
#             open(output_file, 'w', newline='') as write_obj:
#         # Create a csv.reader object from the input file object
#         csv_reader = reader(read_obj,delimiter="\t" )
#         # Create a csv.writer object from the output file object
#         csv_writer = writer(write_obj)
#         # Read each row of the input csv file as list
#         for row in csv_reader:
#             # Pass the list / row in the transform function to add column text for this row
#             transform_row(row, csv_reader.line_num)
#             # Write the updated row / list to the output file
#             csv_writer.writerow(row)
#         datareader = csv_reader(input_file, delimiter="\t")
#         next(datareader,
#                  None)  # https://stackoverflow.com/questions/14257373/skip-the-headers-when-editing-a-csv-file-using-python
#             # tsvfile.seek(0)  # back to the start of the CSV file
#             # https://stackoverflow.com/questions/31537516/loop-through-tsv-csv-file-stops-after-run
#         for columns in datareader:
#             column1, column2, column3, column4, column5, column6 = columns
#             sentence = ((column4), (column5), (column6))
#             sentence = str(sentence)
#             print(sentence)
#
#             ''' Pipeline with CardiffNLP'''
#             model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
#             sentiment_task_cardiff = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
#             print(column1, ' : ')
#             cardiff_sentiment_label = (sentiment_task_cardiff(sentence))
#             print('Cardiff NLP: ', str(sentiment_task_cardiff(sentence)))
#             cardiff_sentiment.append(cardiff_sentiment_label)
#             print(cardiff_sentiment)
#
#             ''' NLP town Model '''  # Instantiate a pipeline object with our task and model passed as parameters,  #https://www.kdnuggets.com/2021/06/create-deploy-sentiment-analysis-app-api.html
#
#             sentiment_task_nlptown = pipeline(task='sentiment-analysis',
#                                               model='nlptown/bert-base-multilingual-uncased-sentiment')
#
#             # Pass the text to our pipeline and print the results
#             nlp_town_sentiment_label= sentiment_task_nlptown(sentence)
#             print('NLP town', f'{sentiment_task_nlptown(sentence)}')
#             nlp_town_sentiment.append(nlp_town_sentiment_label)
#             print(nlp_town_sentiment)
#
#             ''' Using Vader'''  # https://github.com/cjhutto/vaderSentiment
#             # using vader multihttps://pypi.org/project/vader-multi/
#             # https://github.com/brunneis/vader-multi
#
#             analyzer = SentimentIntensityAnalyzer()
#             vader_sentiment_label = analyzer.polarity_scores(sentence)
#             print('Vader: ', f'{str(vader_sentiment_label)}')
#             vader_sentiment.append(vader_sentiment_label)
#             print(vader_sentiment)
#
#
# add_column_in_csv(filename, 'output_3.csv',  lambda row, line_num: row.append(cardiff_sentiment))
# add_column_in_csv(filename, 'output_3.csv',  lambda row, line_num: row.append(nlp_town_sentiment))
# add_column_in_csv(filename, 'output_3.csv',  lambda row, line_num: row.append(va))
#


# with open(filename, 'r',
# #           encoding='latin-1') as tsvfile:  # https://stackoverflow.com/questions/44251813/unicodedecodeerror-charmap-codec-cant-decode-byte-0x9d-in-position-1010494
# #     datareader = csv.reader(tsvfile, delimiter="\t")
# #     next(datareader,
# #          None)  # https://stackoverflow.com/questions/14257373/skip-the-headers-when-editing-a-csv-file-using-python
# #     # tsvfile.seek(0)  # back to the start of the CSV file
# #     # https://stackoverflow.com/questions/31537516/loop-through-tsv-csv-file-stops-after-run
# # for columns in datareader:
# #     column1, column2, column3, column4, column5, column6 = columns
# #     sentence = ((column4), (column5), (column6))
# #     sentence = str(sentence)
# #     print(sentence)
# #
# #     ''' Pipeline with CardiffNLP'''
# #     model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
# #     sentiment_task_cardiff = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
# #     print(column1, ' : ')
# #     cardiff_sentiment_label= (sentiment_task_cardiff(sentence))
# #     print('Cardiff NLP: ', str(sentiment_task_cardiff(sentence)))
# #     cardiff_sentiment.append(cardiff_sentiment)
# #     print(cardiff_sentiment)
# #
# #     ''' NLP town Model '''  # Instantiate a pipeline object with our task and model passed as parameters,  #https://www.kdnuggets.com/2021/06/create-deploy-sentiment-analysis-app-api.html
# #
# #     sentiment_task_nlptown = pipeline(task='sentiment-analysis',
# #                                       model='nlptown/bert-base-multilingual-uncased-sentiment')
# #
# #     # Pass the text to our pipeline and print the results
# #     print('NLP town', f'{sentiment_task_nlptown(sentence)}')
# #
# #     ''' Using Vader'''  # https://github.com/cjhutto/vaderSentiment
# #     # using vader multihttps://pypi.org/project/vader-multi/
# #     # https://github.com/brunneis/vader-multi
# #
# #     analyzer = SentimentIntensityAnalyzer()
# #     vs = analyzer.polarity_scores(sentence)
# #     print('Vader: ', f'{str(vs)}')

#add_column_in_csv(filename, 'output_4.csv', lambda row, line_num: row.append(cardiff_sentiment[line_num - 1])



    #dataframe = pd.read_csv(filename, sep="\t",
                   #        skiprows=4)  # https://thispointer.com/pandas-skip-rows-while-reading-csv-file-to-a-dataframe-using-read_csv-in-python/


    #
        # new_column_name = 'Vader'
        # with open('new_file.tsv' 'w') as outfile:
        #         for line in filename:
        #             if first_line:
        #                 outfile.write('{} {}\n'.format(line, new_column_name))
        #                 first_line = False
        #             else:
        #                 values = line.split()
        #                 if values:
        #                     values.append(values[-1])
        #                 outfile.write(' '.join(values) + '\n')


