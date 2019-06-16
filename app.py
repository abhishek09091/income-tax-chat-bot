from flask import Flask, request, redirect, redirect, jsonify, render_template
import pandas as pd
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import Levenshtein
import atexit
from tqdm import tqdm
from datetime import datetime


import gensim
import logging
import multiprocessing
import os
import bz2
from gensim.models.fasttext import FastText
from gensim.models.word2vec import Word2Vec

import numpy as np
from numpy import dot
from numpy.linalg import norm

import logging
from logging.handlers import RotatingFileHandler


logger = logging.getLogger('log_file.log')
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler('log_file.log', maxBytes=10000, backupCount=5)
logger.addHandler(handler)    



def exit_handler():
    feedback_file_object.close()
    handler.close()
    print('My application is ending!')


atexit.register(exit_handler)

app = Flask(__name__, static_url_path='/static')
app.secret_key = 'CB'


question_vocab = {}
faq_tf_idf = 1
answer_list, original_id = [], []
feedback_file_object = open("feedback_file.txt", "a+")
train_sentence_vector = 1
tf_idf = 1
model = 1

@app.before_first_request
def loader():
    print("One time loading started...")
    global train_sentence_vector, model, question_vocab, faq_tf_idf, answer_list, original_id, tf_idf, feedback_file_object

    faq_data = pd.read_csv("dataset/master_faq.csv")
    print("Training faq data : ", faq_data.shape, faq_data.columns)

    question_list = faq_data['Question'].tolist()
    answer_list = faq_data['Answer'].tolist()
    original_id = faq_data['Original_id'].tolist()

    tf_idf = TfidfVectorizer()
    faq_tf_idf = tf_idf.fit_transform(faq_data['Question'].values.astype('str'))

    # model = FastText.load('./model/fasttext.model')
    # print("Fastetext Model loaded successfully...")


    # train_sentence_vector = []
    # for sentence in tqdm(question_list):
    #     train_sentence_vector.append(get_sentence_vector(str(sentence)))

    # train_sentence_vector = np.array([np.array(xi) for xi in train_sentence_vector])
    # print("Final train vector shape : ", train_sentence_vector.shape)

    for i in range(faq_data.shape[0]):
        ques = str(faq_data['Question'][i])
        for word in ques.split():
            if word in question_vocab:
                question_vocab[word] += 1
            else:
                question_vocab[word] = 1

    print("Vocab length of question is : ", len(question_vocab))


    print("TF-IDF Shape is : ", faq_tf_idf.shape, tf_idf)
    print("One time loading finished..")


def leveinstein_on_word(word):
    new_word = [""]
    max_count = 0
    min_distance = 999
    for ques_word_key, ques_word_value in question_vocab.items():
        distance = Levenshtein.distance(word, ques_word_key)
        if distance < min_distance:
            new_word[0] = ques_word_key
            min_distance = distance
            max_count = ques_word_value
        elif distance == min_distance:
            if max_count < ques_word_value:
                max_count = ques_word_value
                new_word[0] = ques_word_key
    return new_word[0], min_distance, max_count


def leveinstein_on_query(query):
    print("Query before lev dis function : ", query)
    after_lev_dis = []
    for word in query.split():
        if word in question_vocab:
            after_lev_dis.append(word)
        else:
            new_word, dist, count = leveinstein_on_word(word)
            print("Old Word : {}  , New Word : {} , Distance : {}, Count : {}".format(word, new_word, dist, count))
            if dist <= 2:
                after_lev_dis.append(new_word)
    print("Query after lev dis function : ", " ".join(after_lev_dis))
    return " ".join(after_lev_dis)


def get_sentence_vector(sentence):
    # sentence = str(sentence)
    sent_vec = np.zeros(100)
    numw = 0
    try:
        temp_tf_idf = tf_idf.transform([sentence])
    except IOError:
        _, value, traceback = sys.exc_info()
        print('Error opening %s: %s' % (value.filename, value.strerror))
    for w in sentence.split():
        try:
            sent_vec = np.add(sent_vec, temp_tf_idf[0, tf_idf.vocabulary_[w]] * model[w])
            numw += 1
        except:
            pass
    return sent_vec / np.sqrt(sent_vec.dot(sent_vec))


@app.route('/send_query_response', methods=['POST'])
def query_response():
    """
    Query response using TF-IDF score
    """
    if request.method == 'POST':
        query = request.form['query']
        query = query.lower()
        logger.warning(str(datetime.now()) + " | " + query + " | " + "TF-IDF(0.6)")
        query = leveinstein_on_query(query)
        print(query)
        query_tf_idf = tf_idf.transform([query])
        cosine_similarities = cosine_similarity(query_tf_idf, faq_tf_idf).flatten()
        related_docs_indices = cosine_similarities.argsort()[::-1][:2]
        if cosine_similarities[related_docs_indices[0]] < 0.6:
            query_ans, _id, confidence_score = "Sorry I am unable to answer that. Can you rephrase the question.", -1, round(cosine_similarities[related_docs_indices[0]], 2)
        else:
            query_ans, _id, confidence_score = answer_list[related_docs_indices[0]], original_id[related_docs_indices[0]], round(cosine_similarities[related_docs_indices[0]], 2)
        return jsonify({'query_ans': query_ans, 'id': _id, 'confidence_score':confidence_score})
    else:
        return jsonify({"Call": "GET not allowed.."})


# @app.route('/send_query_response', methods=['POST'])
# def query_response():
#     """
#     Query response using Cosine similarity using word embeddings
#     """
#     if request.method == 'POST':
#         query = request.form['query']
#         query = query.lower()
#         logger.warning(str(datetime.now()) + " | " + query + " | " + "FASTTEXT(0.8)")
#         query_vec = get_sentence_vector(query)
#         query_vec = np.asarray(query_vec)

#         similarity_score = np.matmul(train_sentence_vector, query_vec)
#         similarity_score_index = np.argsort(similarity_score)[::-1][:2]

#         if similarity_score[similarity_score_index[0]] < 0.6:
#             query_ans, _id, confidence_score = "Sorry I am unable to answer that. Can you rephrase the question again.", -1, round(similarity_score[similarity_score_index[0]], 2)
#         else:
#             query_ans, _id, confidence_score = answer_list[similarity_score_index[0]], original_id[similarity_score_index[0]], round(similarity_score[similarity_score_index[0]], 2)
#             print(({'status':200, 'query_ans': query_ans, 'id': _id, 'confidence_score': confidence_score}))
#         return jsonify({'status':200, 'query_ans': query_ans, 'id': _id, 'confidence_score': confidence_score})
#     else:
#         return jsonify({"Call": "GET not allowed.."})


@app.route('/get_query_feedback', methods=['POST'])
def query_feedback():
    if request.method == 'POST':
        try:
            feedback = request.form['feedback']
            answer = request.form['answer']
            query = request.form['query']
        except KeyError:
            print("Key Error for feedback, id, query..")
            return jsonify({'status': -1, 'error': 'Key Error'})
        feedback_string = str(feedback) + "|" + str(answer) + "|" + str(query) + "\n"
        feedback_file_object.write(feedback_string)
        return jsonify({'stat': 200, 'error': 'No Error', 'call': 'Thanks for sharing the feedback.'})



@app.route('/test', methods=['POST'])
def test():
    if request.method == 'POST':
        return jsonify({'stat': '200'})


@app.route('/')
def home():
    return render_template('home.html', call='')


TEMPLATES_AUTO_RELOAD = True


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)


# if __name__ == '__main__':
#     app.run(debug=True)
