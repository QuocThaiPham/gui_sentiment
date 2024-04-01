import pandas as pd
import numpy as np
from underthesea import word_tokenize, pos_tag, sent_tokenize
import re
import demoji
from pyvi import ViPosTagger, ViTokenizer
import string


# Define Function to preprocess
def process_text(text, emoji_dict, teen_dict, wrong_lst,eng_vn_dict):
    document = text.lower()
    document = document.replace("’", '')
    document = re.sub(r'\.+', ".", document)
    new_sentence = ''
    for sentence in sent_tokenize(document):
        # CONVERT EMOJI ICON
        sentence = ''.join(' ' + emoji_dict[word] + ' ' if word in emoji_dict else word for word in list(sentence))

        # CONVERT TEEN CODE
        sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())

        # CONVERT ENGLISH CODE
        sentence = ' '.join(eng_vn_dict[word] if word in eng_vn_dict else word for word in sentence.split())

        # DEL Punctuation & Numbers
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(re.findall(pattern, sentence))

        ###### DEL wrong words
        sentence = ' '.join('' if word in wrong_lst else word for word in sentence.split())
        new_sentence = new_sentence + sentence + '. '
        document = new_sentence

        # DEL excess blank space
        document = re.sub(r'\s+', ' ', document).strip()


    return document


def process_special_word(text):
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
            new_text = new_text + word + ' '
    else:
        new_text = text
    return new_text.strip()


# Hàm để chuẩn hóa các từ có ký tự lặp
def normalize_repeated_characters(text):
    # Thay thế mọi ký tự lặp liên tiếp bằng một ký tự đó
    # Ví dụ: "ngonnnn" thành "ngon", "thiệtttt" thành "thiệt"
    return re.sub(r'(.)\1+', r'\1', text)


def preprocess_postag_thesea(text , stop_lst):
    document = text.lower()
    document = document.replace("’", '')
    document = re.sub(r'\.+', ".", document)

    new_document = ''
    for sentence in sent_tokenize(document):
        sentence = sentence.replace('.', '')

        # POS tag
        lst_word_type = ['N', 'Np', 'A', 'AB', 'V', 'VB', 'VY', 'R']

        # lst_word_type = ['A','AB','V','VB','VY','R']
        sentence = ' '.join(
            word[0] if word[1].upper() in lst_word_type else '' for word in pos_tag(
                process_special_word(word_tokenize(sentence=sentence,
                                                   format='text')
                                     )
            )
        )
        new_document = new_document + sentence + ' '

    # DEL excess blank space
    new_document = re.sub(r'\s+', ' ', new_document).strip()

    # Remove duplicated characters
    new_document = normalize_repeated_characters(text=new_document)

    # Remove stop words
    new_document = remove_stopword(text=new_document,stopwords=stop_lst)
    return new_document


def remove_stopword(text, stopwords):
    # REMOVE stop words
    document = ' '.join('' if word in stopwords else word for word in text.split())
    # print(document)
    # DEL excess blank space
    document = re.sub(r'\s+', ' ', document).strip()
    return document


def find_words(document, list_of_words):
    document_lower = document.lower()
    word_count = 0
    word_list = []
    for word in list_of_words:
        if word in document_lower:
            print(word)
            word_count += document_lower.count(word)
            word_list.append(word)
    return word_count, word_list
