import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
import numpy as np
from underthesea import word_tokenize, pos_tag, sent_tokenize, sentiment
from tqdm.auto import tqdm
from collections import defaultdict, Counter


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


def create_adj_wordcloud(df,cleanser):
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
    return full_adj_word, wc_pos


def restaurant_sentiment_analysis(final_df, id_res=None, top_words: int = 5):
    global res_df
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

        print(">>> OVERVIEW DATA")
        print("Restaurant name:", restaurant_name)
        print("Type restaurant:", type_restaurant)
        print("Average Rating:", round(rating_score, 2))
        print("Average Price:", price)
        print("Address:", address)

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

                word_ls, pos_wc = create_adj_wordcloud(df=df)
                print("--- Succeed to create WordCloud")
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

        # DISPLAY WORDCLOUD
        fig = plt.figure(figsize=(15, 7))
        # setting values to rows and column variables
        columns = len(wc_ls)
        for i in range(columns):
            # Title name:
            name = wc_ls[i]['name']
            # Adds a subplot at the 1st position
            fig.add_subplot(1, columns, i + 1)

            # showing image
            plt.imshow(wc_ls[i]['wordcloud'])
            plt.axis('off')
            plt.title(f"WordCloud of {name} review")

        # DISPLAY TOP WORD in each class
        print("\n\n>>> GET TOP 5 WORDS OCCURING MOST IN EACH CLASS:\n")
        for kind in full_word_ls:
            name = kind['name']
            concat_list = get_top_duplicated_words(kind['words'], top_n=top_words)
            print(f'{name}: {concat_list}')

        # DISPLAY RATE OF EACH TYPE REVIEW BY MONTH
        palette = sns.color_palette('husl', n_colors=3)
        fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        sns.lineplot(x=df_pos['date_review'].dt.month,
                     y=df_pos['rating_scaler'],
                     color=palette[0],
                     label='Positive',
                     ax=ax[0])
        ax[0].set_ylabel("Rating")

        sns.lineplot(x=df_neg['date_review'].dt.month,
                     y=df_neg['rating_scaler'],
                     color=palette[1],
                     label='Negative',
                     ax=ax[1])
        ax[1].set_ylabel("Rating")

        sns.lineplot(x=df_neu['date_review'].dt.month,
                     y=df_neu['rating_scaler'],
                     color=palette[2],
                     label='Neutral',
                     ax=ax[2])
        ax[2].set_ylabel("Rating")

        # Set a big title for the entire figure
        fig.suptitle('Rating of Type Line Thourgh by Month', fontsize=16, y=1.02)
        plt.xlabel('Review Month')
        fig.tight_layout()
        plt.show()
