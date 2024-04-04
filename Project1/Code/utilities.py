import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
import numpy as np
from underthesea import word_tokenize, pos_tag, sent_tokenize, sentiment
from tqdm.auto import tqdm
from collections import defaultdict, Counter
import streamlit as st


def get_dict_word(file_path: str) -> dict:
    with open(file_path, 'r', encoding="utf8") as file:
        _lst = file.read().split('\n')
        _dict = {}
        for line in _lst:
            key, value = line.split('\t')
            _dict[key] = str(value)

    return _dict


def preprocessing_text(text,
                       cleanser,
                       vector_model,  # ='resources/countvectorizer_model.pkl',
                       tfidf_model):  # 'resources/tfidf_model.pkl'
    clean_text = cleanser.process_text(text=text)
    clean_text = cleanser.convert_unicode(str(clean_text))
    clean_text = cleanser.process_postag_thesea(clean_text)
    a = cleanser.extract_adjectives_vietnamese(text)
    print(a)
    # initialize count vectorized and TFIDF
    vectorizer = vector_model
    tfidf = tfidf_model

    # transform X_test
    _pre_X_test = vectorizer.transform([clean_text])
    _pre_X_test = tfidf.transform(_pre_X_test)

    return _pre_X_test


def display_emotion(emotion_index,
                    like_path="resources/like_icon.png",
                    not_like_path="resources/not_like_icon.png",
                    neutral_path="resources/neutral_icon.png"):
    if emotion_index == 0:
        st.image(not_like_path, caption="Not Like", width=200)
    elif emotion_index == 1:
        st.image(neutral_path, caption="Neutral", width=200)
    elif emotion_index == 2:
        st.image(like_path, caption="Like", width=200)


def get_top_duplicated_words(word_list, top_n=5):
    word_count_dict = Counter(word_list)

    # Get the top N duplicated words
    top_duplicated_words = word_count_dict.most_common(top_n)

    return top_duplicated_words


# def extract_adjectives_vietnamese(comment):
#     # Perform part-of-speech tagging
#     tagged_words = pos_tag(comment)
#
#     # Extract adjectives
#     adjectives = [word for word, pos in tagged_words if pos == 'A']
#
#     return adjectives


def create_adj_wordcloud(df, cleanser,title=''):
    type_comment = df['result'].head(1).values[0].upper()
    full_adj_word = []
    for k, text in tqdm(df['clean_review'].items(), f"Extract Adjective word from {type_comment}"):
        if isinstance(text, str):
            word_ls = cleanser.extract_adjectives_vietnamese(text)
            full_adj_word.extend(word_ls)

    _comments = ' '.join(full_adj_word)
    # print(_comments)
    # Generate word cloud from comments
    wc_pos = WordCloud(background_color='white',
                       collocations=False,
                       max_words=50).generate(_comments)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Display WordCloud using Matplotlib's imshow on the specified ax
    ax.imshow(wc_pos, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    # Display WordCloud in Streamlit
    st.pyplot(fig)
    return full_adj_word, wc_pos


def restaurant_sentiment_analysis(final_df,
                                  res_df,
                                  cleanser,
                                  id_res=None,
                                  top_words: int = 5):
    # ---- FIRST STEP: SELECT RESTAURANT ---
    if id_res:
        id_selected = id_res
    else:
        name_selected = input("Please type keyword of restaurant:")
        filter_ = res_df['Restaurant'].str.findall(rf'{name_selected}').apply(lambda x: len(x) > 0)

        with pd.option_context('display.max_rows', 30,
                               'display.max_columns', 10,
                               'display.precision', 3,
                               'display.width', None,
                               'display.max_colwidth', None
                               ):
            print(res_df.loc[filter_, ['ID', 'Restaurant', 'street', 'ward', 'district']].to_string())

        # Select ID from above restaurant
        id_selected = int(input("Please select restaurant ID from above restaurants:"))

    # Validate the restaurant
    validation_res = res_df['ID'] == id_selected
    if not any(validation_res):
        print("Dont find this restaurant, please choose again")
        return None
    else:
        # OVERVIEW DATA
        # Extract ratings and comments
        selected_res = final_df[final_df['ID'] == id_selected]
        restaurant_name = selected_res['Restaurant'].values[0]
        type_restaurant = selected_res['type_restaurant'].values[0]
        rating_score = selected_res['rating_scaler'].values[0]
        price = selected_res['avg_price'].values[0]
        address = f"{selected_res['street'].values[0]} {str(selected_res['ward'].values[0])} District {selected_res['district'].values[0]}"

        st.write(f"""
        ***>>> OVERVIEW DATA***
        - Restaurant name: {restaurant_name}
        - Type restaurant: {type_restaurant}
        - Average Rating:: {round(rating_score, 2)}
        - Average Price:: {price}
        - Address:: {address}
""")

        # Extract Positive Negative and Neutral
        df_pos = selected_res[selected_res['result'] == 'positive']
        df_neg = selected_res[selected_res['result'] == 'negative']
        df_neu = selected_res[selected_res['result'] == 'neutral']

        wc_ls = []
        full_word_ls = []
        for i, df in enumerate([df_pos, df_neg, df_neu]):
            if df.shape[0] > 0:
                if i == 0:
                    name = 'POSITIVE'
                elif i == 1:
                    name = 'NEGATIVE'
                else:
                    name = 'NEUTRAL'

                st.subheader(f'{name} Review')
                word_ls, pos_wc = create_adj_wordcloud(df=df,
                                                       cleanser=cleanser)
                wc_ls.append({'name': name, 'wordcloud': pos_wc})
                full_word_ls.append({'name': name, 'words': word_ls})
            else:
                if i == 0:
                    name = 'POSITIVE'
                elif i == 1:
                    name = 'NEGATIVE'
                else:
                    name = 'NEUTRAL'
                print(f"\n>>> There is not type **{name}** in DataFrame")

        # DISPLAY TOP WORD in each class
        st.subheader("GET TOP 5 WORDS OCCURRING MOST IN EACH CLASS:")
        for kind in full_word_ls:
            name = kind['name']
            concat_list = get_top_duplicated_words(kind['words'], top_n=top_words)
            st.write(f'### {name}')
            text = '\n'.join([f"- {v[0]}: {v[1]} words" for v in concat_list])
            st.write(text)

        # DISPLAY RATE OF EACH TYPE REVIEW BY MONTH
        st.subheader('Rating of Type Line Thourgh by Month')
        palette = sns.color_palette(palette='husl', n_colors=3)
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(16, 15), sharex=True)
        sns.lineplot(x=df_pos['date_time'].dt.month,
                     y=df_pos['rating_scaler'],
                     color=palette[0],
                     label='Positive',
                     ax=ax[0])
        ax[0].set_ylabel("Rating",fontsize=16)
        ax[0].tick_params(axis='x', labelsize=15)  # Adjust font size of x-values
        ax[0].tick_params(axis='y', labelsize=15)  # Adjust font size of y-values
        ax[0].legend(fontsize=20)

        sns.lineplot(x=df_neg['date_time'].dt.month,
                     y=df_neg['rating_scaler'],
                     color=palette[1],
                     label='Negative',
                     ax=ax[1])
        ax[1].set_ylabel("Rating",fontsize=16)
        ax[1].tick_params(axis='x', labelsize=15)  # Adjust font size of x-values
        ax[1].tick_params(axis='y', labelsize=15)  # Adjust font size of y-values
        ax[1].legend(fontsize=20)

        sns.lineplot(x=df_neu['date_time'].dt.month,
                     y=df_neu['rating_scaler'],
                     color=palette[2],
                     label='Neutral',
                     ax=ax[2])
        ax[2].set_ylabel("Rating",fontsize=16)
        ax[2].tick_params(axis='x', labelsize=15)  # Adjust font size of x-values
        ax[2].tick_params(axis='y', labelsize=15)  # Adjust font size of y-values
        ax[2].legend(fontsize=20)

        # Set a big title for the entire figure
        plt.xlabel('Review by Month',fontsize=20)
        fig.tight_layout()
        st.pyplot(fig)
