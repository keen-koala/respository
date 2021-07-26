from csv import reader, writer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import csv
import re


filename = "C:/Users/Vanessa/Desktop/Masterarbeit/MA_repository/respository/idiom_sentences_multilingual_test.tsv"
delete_character_lits=["</s><s>", "'", ]


'''Old pipeline '''

with open(filename, 'r', encoding='utf-8') as tsvfile:  # https://stackoverflow.com/questions/44251813/unicodedecodeerror-charmap-codec-cant-decode-byte-0x9d-in-position-1010494
    datareader = csv.reader(tsvfile, delimiter="\t")
    next(datareader,None)  # https://stackoverflow.com/questions/14257373/skip-the-headers-when-editing-a-csv-file-using-python
    #tsvfile.seek(0)  # back to the start of the CSV file # https://stackoverflow.com/questions/31537516/loop-through-tsv-csv-file-stops-after-run
    for columns in datareader:
        #next(tsvfile)
        column1, column2, column3, column4, column5, column6 = columns
        sentence = ((column4), (column5), (column6))
        sentence = str(sentence)
        index= column1
        cleaned_sentence= sentence.replace("</s><s>"," ",  '')
        sentence_final= cleaned_sentence.replace("'", '')
        print(index, ': ', sentence_final)

        ''' Pipeline with CardiffNLP'''
        model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        sentiment_task_cardiff = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
        cardiff_sentiment_label = (sentiment_task_cardiff(sentence_final))
        print('Cardiff NLP: ', str(sentiment_task_cardiff(sentence_final)))


        ''' NLP town Model '''  # Instantiate a pipeline object with our task and model passed as parameters,  #https://www.kdnuggets.com/2021/06/create-deploy-sentiment-analysis-app-api.html

        sentiment_task_nlptown = pipeline(task='sentiment-analysis',
                                          model='nlptown/bert-base-multilingual-uncased-sentiment')
        nlp_town_sentiment_label= sentiment_task_nlptown(sentence_final)

        # Pass the text to our pipeline and print the results
        print('NLP town', f'{sentiment_task_nlptown(sentence_final)}')

        ''' Using Vader'''  # https://github.com/cjhutto/vaderSentiment
        # using vader multihttps://pypi.org/project/vader-multi/
        # https://github.com/brunneis/vader-multi

        analyzer = SentimentIntensityAnalyzer()
        vs = analyzer.polarity_scores(sentence_final)
        print('Vader: ', vs)














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


