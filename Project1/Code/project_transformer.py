# SET UP BASE ESTIMATOR
import sklearn.compose as sc
import sklearn.preprocessing as sp
from sklearn.base import TransformerMixin, BaseEstimator
from imblearn.pipeline import Pipeline

# TRAIN TEST DATASET
from sklearn.model_selection import train_test_split
# HANDLE DATAFRAME
import pandas as pd
import numpy as np
import os
from datetime import date as dt
import warnings

from underthesea import word_tokenize, pos_tag, sent_tokenize
import re

# SET UP BASE ESTIMATOR
import sklearn.compose as sc
import sklearn.preprocessing as sp

from collections import defaultdict
from sklearn.base import TransformerMixin, BaseEstimator
from imblearn.pipeline import Pipeline


# CREATE CLASS FOR CUSTOM TRANSFORMER
class Data_Wrangling(BaseEstimator, TransformerMixin):
    def __init__(self, emoji_dict, teen_dict, wrong_lst, eng_vn_dict, stop_words):
        self.emoji_dict = emoji_dict
        self.teen_dict = teen_dict
        self.wrong_lst = wrong_lst
        self.eng_vn_dict = eng_vn_dict
        self.stop_words = stop_words

    def remove_stopword(self, text):
        # REMOVE stop words
        document = ' '.join('' if word in self.stop_words else word for word in text.split())
        # print(document)
        # DEL excess blank space
        document = re.sub(r'\s+', ' ', document).strip()
        return document

    def loaddicchar(self):
        dic = {}
        char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
            '|')
        charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
            '|')
        for i in range(len(char1252)):
            dic[char1252[i]] = charutf8[i]
        return dic

    # Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
    def convert_unicode(self,txt):
        dicchar = self.loaddicchar()
        return re.sub(
            r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
            lambda x: dicchar[x.group()], txt)

    def process_text(self, text):
        document = text.lower()
        document = document.replace("’", '')
        document = re.sub(r'\.+', "", document)
        new_sentence = ''
        for sentence in sent_tokenize(document):
            sentence = ''.join(' ' + self.emoji_dict[word] + ' ' if word in self.emoji_dict else word for word in list(sentence))  # CONVERT EMOJI ICON
            sentence = ' '.join(self.teen_dict[word] if word in self.teen_dict else word for word in sentence.split())  # CONVERT TEEN CODE
            sentence = ' '.join(self.eng_vn_dict[word] if word in self.eng_vn_dict else word for word in sentence.split())  # CONVERT ENGLISH CODE

            # DEL Punctuation & Numbers
            pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
            sentence = ' '.join(re.findall(pattern, sentence))

            # DEL wrong words
            sentence = ' '.join('' if word in self.wrong_lst else word for word in sentence.split())

            # Remove stop words
            sentence = self.remove_stopword(text=sentence)
            new_sentence = new_sentence + sentence
            document = new_sentence

            # DEL excess blank space
            document = re.sub(r'\s+', ' ', document).strip()
        return document

    def process_special_word(self, text):
        # có thể có nhiều từ đặc biệt cần ráp lại với nhau
        new_text = ''
        text_lst = text.split()
        i = 0
        # không, chẳng, chả...
        if 'không' in text_lst:
            while i <= len(text_lst) - 1:
                word = text_lst[i]
                # print(word)
                # print(i)
                if word == 'không':
                    next_idx = i + 1
                    if next_idx <= len(text_lst) - 1:
                        word = word + '_' + text_lst[next_idx]
                    i = next_idx + 1
                else:
                    i = i + 1
                word = re.sub(r'(.)\1+', r'\1', word)
                new_text = new_text + word + ' '
        else:
            new_text = text
        return new_text.strip()

    def extract_adjectives_vietnamese(self, comment: str):
        # Perform part-of-speech tagging
        tagged_words = pos_tag(comment)

        # Extract adjectives
        adjectives = [word for word, pos in tagged_words if pos == 'A']

        return adjectives

    def create_word_count_dictionary(self, word_list):
        word_count_dict = defaultdict(int)

        # Count the occurrences of each word
        for word in word_list:
            word_count_dict[word] += 1

        # Convert the default-dict to a regular dictionary
        word_count_dict = dict(word_count_dict)

        return word_count_dict

    def process_postag_thesea(self, text):
        new_document = ''
        for sentence in sent_tokenize(text):
            sentence = sentence.replace('.', '')

            # Remove punctuation symbols
            punctuation_pattern = r'[^\w\s]'  # Matches any character that is not a word character or whitespace
            sentence = re.sub(punctuation_pattern, '', sentence)
            # ---- POS tag
            lst_word_type = ['N', 'Np', 'A', 'AB', 'V', 'VB', 'VY', 'R']
            # lst_word_type = ['A','AB','V','VB','VY','R']
            sentence = ' '.join(word[0] if word[1].upper() in lst_word_type else '' for word in
                                pos_tag(self.process_special_word(word_tokenize(sentence, format="text"))))
            new_document = new_document + sentence + ' '

        # DEL excess blank space
        new_document = re.sub(r'\s+', ' ', new_document).strip()
        return new_document

    def fit(self, X, y=None):
        print('\n>>>>Data_Wrangling.fit() called.')
        self.X = X
        return self

    def transform(self, X, y=None):
        print('>>>>Data_Wrangling.transform() called.')
        self.X = X
        try:
            pass

        except Exception as e:
            # Raise error
            # raise e
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(e).__name__, e.args)
            print(message)
            return message

        else:
            print('>>>>Finish Data_Wrangling.transform().\n')

            return self.X
