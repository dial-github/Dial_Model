# coding=utf-8
from __future__ import unicode_literals
import gc
import io
import json
from our import DIAL
import pandas as pd
from bs4 import BeautifulSoup
import re
import numpy as np
from sklearn.model_selection import train_test_split
from nltk import tokenize
from keras.utils.np_utils import to_categorical
import operator
import codecs
import os
from keras import backend as K
import argparse

# when gpu>1 choose a gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
DATA_PATH = 'data'
SAVED_MODEL_DIR = 'saved_models'
SAVED_MODEL_FILENAME = 'DIAL_model.h5'
EMBEDDINGS_PATH = 'saved_models/glove.6B.100d.txt'


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


if __name__ == '__main__':
    # dataset used for training
    parser = argparse.ArgumentParser()
    
    args = parser.parse_args()

    if args.platform:
        platform = args.platform
    
    data_train = pd.read_csv('data/' + platform + '_content_no_ignore.tsv', sep='\t')
    VALIDATION_SPLIT = 0.25
    contents = []
    labels = []
    texts = []
    ids = []

    for idx in range(data_train.content.shape[0]):
        text = BeautifulSoup(data_train.content[idx], features="lxml")
        text = clean_str(text.get_text().encode('ascii', 'ignore'))
        texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        contents.append(sentences)
        ids.append(data_train.id[idx])

        labels.append(data_train.label[idx])

    labels = np.asarray(labels)
    labels = to_categorical(labels)

    # load user comments
    comments = []
    comments_text = []
    comments_train = pd.read_csv('data/' + platform + '_comment_no_ignore.tsv', sep='\t')


    content_ids = set(ids)

    for idx in range(comments_train.comment.shape[0]):
        if comments_train.id[idx] in  content_ids:
            com_text = BeautifulSoup(comments_train.comment[idx], features="lxml")
            com_text = clean_str(com_text.get_text().encode('ascii', 'ignore'))
            tmp_comments = []
            for ct in com_text.split('::'):
                if ct:
                    tmp_comments.append(ct)
            comments.append(tmp_comments)
            comments_text.extend(tmp_comments)
    comments_id = []
    with io.open('./data/'+ platform + 'userid.json', 'r', encoding='utf-8') as f:
        commnets_dict = json.load(f)
        comments_id_train = pd.read_csv('data/' + platform + '_comment_id_no_ignore.tsv', sep='\t')
        for idx in range(comments_id_train.comment.shape[0]):
            if comments_id_train.id[idx] in  content_ids:
                com_text = BeautifulSoup(comments_id_train.comment[idx], features="lxml")
                com_text = clean_str(com_text.get_text().encode('ascii', 'ignore'))
                tmp_comments_id = []
                for ct in com_text.split('::'):
                    if ct:
                        tmp_comments_id.append(commnets_dict[ct])
                comments_id.append(tmp_comments)
    
    id_train, id_test, x_train, x_val, y_train, y_val, c_train, c_val, cid_train, cid_val = train_test_split(ids,contents, labels, comments, comments_id,
                                                                      test_size=VALIDATION_SPLIT, random_state=42,
                                                                      stratify=labels)
    
    
    # Train and save the model
    batch_size = 20
    SAVED_MODEL_FILENAME = platform + '_DIAL_new_model.h5'
    
    h = DIAL(platform, alter_params='null')

    h.train(x_train, y_train, c_train, cid_train, cid_val, c_val, x_val, y_val,
                    batch_size=batch_size,
                    epochs=30,
                    embeddings_path='./glove.6B.100d.txt',
                    saved_model_dir=str(SAVED_MODEL_DIR),
                    saved_model_filename=str(SAVED_MODEL_FILENAME))
    

    h.load_weights(saved_model_dir = str(SAVED_MODEL_DIR), saved_model_filename = str(SAVED_MODEL_FILENAME))
    