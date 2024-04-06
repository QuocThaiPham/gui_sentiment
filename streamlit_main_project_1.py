import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
from Project1.Code.project_transformer import Data_Wrangling
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import io
from contextlib import redirect_stdout
import re
# from Viet_lib.Viet_lib import process_text, convert_unicode, process_postag_thesea, remove_stopword
from Project1.Code.utilities import preprocessing_text, display_emotion, get_top_duplicated_words, create_adj_wordcloud, restaurant_sentiment_analysis, get_dict_word

# ----------------- INITIALIZATION -----------------
emoji_dict = get_dict_word(file_path='Project1/DATA_shopeefood/files/emojicon.txt')  # Emoji
teen_dict = get_dict_word(file_path='Project1/DATA_shopeefood/files/teencode.txt')  # Teen
eng_dict = get_dict_word(file_path='Project1/DATA_shopeefood/files/english-vnmese.txt')  # eng

with open('Project1/DATA_shopeefood/files/wrong-word.txt', 'r', encoding="utf8") as file:  # WRONG WORD
    wrong_lst = file.read().split('\n')

with open('Project1/DATA_shopeefood/files/vietnamese-stopwords.txt', 'r', encoding="utf8") as file:  # STOP WORD
    stop_lst = file.read().split('\n')

# GET CLEANSER
cleanser = Data_Wrangling(emoji_dict=emoji_dict,
                          teen_dict=teen_dict,
                          wrong_lst=wrong_lst,
                          eng_vn_dict=eng_dict,
                          stop_words=stop_lst)
# Model Vectorize and TFIDF
with open('resources/countvectorizer_model.pkl', 'rb') as f:
    vec_m = pickle.load(f)
with open('resources/tfidf_model.pkl', 'rb') as f_2:
    tfidf_m = pickle.load(f_2)


# ----------------- ADD CACHE -----------------
@st.cache_resource()
def get_prd_df(file_path, use_cols):
    df = pd.read_csv(filepath_or_buffer=file_path,
                     usecols=use_cols)
    return df


# res_df = get_prd_df(file_path='Project1/Clean_data/clean_restaurant.csv', use_cols=None)
clean_review_df = get_prd_df(file_path='Project1/Clean_data/clean_review_data.csv', use_cols=None)
final_df = get_prd_df(file_path='Project1/Clean_data/combine_review_res.csv', use_cols=None)
final_df['date_time'] = pd.to_datetime(final_df['date_time'])

# positive_review_df = final_df[final_df['result'] == 'positive']
# negative_review_df = final_df[final_df['result'] == 'negative']
# neutral_review_df = final_df[final_df['result'] == 'neutral']

feat = ['ID', 'Restaurant', 'street', 'ward', 'district', 'start_time',
        'end_time', 'IDRestaurant', 'avg_price', 'type_restaurant']
res_df = final_df[feat]


# ----------------- BUILD GUI -----------------
def fn_business_objective():
    # Upload file
    # uploaded_file = st.file_uploader("Choose a file", type=['csv'])
    # if uploaded_file is not None:
    #     data = pd.read_csv(uploaded_file, encoding='latin-1')
    #     data.to_csv("resources/new_file.csv", index=False)

    st.subheader("EDA DATA")
    st.write("- There are 2 datasets including the restaurant dataframe and the review dataframe")
    st.subheader(":blue[1. Review Dataset]")
    st.write("#### 1.1. View data")
    st.dataframe(clean_review_df.head())
    st.write("There are 29959 reviews")

    st.write("#### 1.2. Plot of Rating Distribution")

    # Display the plot in Streamlit
    sns.set(style="whitegrid")
    fig_hist, ax = plt.subplots()
    sns.histplot(data=clean_review_df,
                 x='rating_scaler',
                 kde=True,
                 color="skyblue",
                 edgecolor='white', ax=ax)
    st.pyplot(fig_hist)

    # st.image('resources/rating_plot.png')
    st.write("""
    - Rating score is scaled from 0 to 10
    - Regards to histogram, the most popular range is from 8 to 10 
    """)

    st.write("#### 1.3. Distribution of Label: Not like - Neutral - Like")
    # Display the plot in Streamlit
    sns.set(style="whitegrid")
    fig_label, ax = plt.subplots()
    label_df = clean_review_df[['label']].replace(to_replace={0: 'Dislike', 1: 'Like', 2: 'Neutral'})
    sns.countplot(data=label_df,
                  x='label',
                  # color="skyblue",
                  edgecolor='white',
                  ax=ax)
    st.pyplot(fig_label)

    # st.image('resources/label_count.png')
    st.write("- As can be seen the count plot, **'Like'** label has the most distribution")

    st.write("#### 1.4. Wordcloud of **Dislike** label")
    # create_adj_wordcloud(df=negative_review_df, cleanser=cleanser)
    st.image('resources/Image/dislike.png')

    st.write("#### 1.5. Wordcloud of **Like** label")
    # create_adj_wordcloud(df=positive_review_df, cleanser=cleanser)
    st.image('resources/Image/Like.png')

    st.write("#### 1.6. Wordcloud of **Neutral** label")
    # create_adj_wordcloud(df=neutral_review_df, cleanser=cleanser)
    st.image('resources/Image/neutral.png')

    st.subheader(":blue[2. Restaurant Dataset]")
    st.write(" #### 2.1. View data")
    st.dataframe(res_df.head(10))

    st.text("DataFrame Info:")
    # Create a StringIO object to capture printed output
    string_buffer = io.StringIO()
    # Redirect printed output to the StringIO object
    with redirect_stdout(string_buffer):
        # Your code that prints data
        res_df.info()
    # Get the captured output as a string
    captured_output = string_buffer.getvalue()

    st.text(f"""{captured_output}""")

    st.write("- There are 1605 restaurants")

    st.write("#### 2.2. District Distribution of All Restaurants]")
    sns.set(style="whitegrid")
    fig_dist_hist, ax = plt.subplots(figsize=(12,12))
    sns.barplot(data=res_df,
                y='district',
                x='avg_price',
                orient='h',
                edgecolor='white',
                ax=ax)
    st.pyplot(fig_dist_hist)

    st.write(""" As can be seen the bar graph,
    * there are many fancy restaurants focusing Dis.2, 12, 3. 
    * There are many casual restaurants in Dis 9, 10, 11.
    """)


def fn_new_prediction():
    # load model classication
    pkl_filename = "resources/project_1_model_SVC.sav"
    model = pickle.load(open(pkl_filename, 'rb'))
    st.subheader(":blue[Upload data or Input data?]")

    type = st.radio("", options=("Input", "Upload"))
    st.write("""
    ##### Example:""")
    st.write('Like: Quán này ngon lắm đó mọi người!!!')
    st.write('Neutral: Quán này tạm ổn thôi à.')
    st.write('Not like: Mọi người nên tránh xa quán này nha, dở lắm.')
    if type == "Upload":
        # Upload file

        # SAMPLE
        st.write("#### 📝 **Please create a CSV file with the following format:** 📄")
        data = {'Content': ['Quán này ngon lắm đó mọi người!!!',
                            'Quán này tạm ổn thôi à.',
                            'Mọi người nên tránh xa quán này nha, dở lắm.']}
        df = pd.DataFrame(data)
        # Display the DataFrame as a table
        st.table(df)

        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            lines_df = pd.read_csv(uploaded_file_1, header=0,delimiter=';')
            st.write("Original DataFrame:")
            st.write(lines_df)
            X_pre = lines_df['Content'].apply(lambda x: preprocessing_text(text=x,
                                                                           cleanser=cleanser,
                                                                           vector_model=vec_m,
                                                                           tfidf_model=tfidf_m))
            labels_dict = {0: 'Dislike', 1: 'Neutral', 2: 'Like'}
            labels = []
            for i,x in enumerate(X_pre):
                y_pred_new = model.predict(x)
                # st.code("New predictions (0: Not Like, 1: Neutral, 2: Like): " + str(y_pred_new))
                st.write(f"Comment: {lines_df.loc[i,['Content']].values[0]}")
                display_emotion(y_pred_new)
                labels.append(labels_dict[y_pred_new.item()])

            # prediction_ = model.predict(np.array([X_pre]))
            # #print(prediction_)
            lines_df['Prediction'] = labels

            st.success("Contents are predicted successfully!")
            st.write("Updated DataFrame:")
            st.write(lines_df)

    if type == "Input":
        st.write("### :blue[Please Input your comment 👇]")
        content = st.text_area(label="")

        if content != "":
            x_new = preprocessing_text(text=content,
                                       cleanser=cleanser,
                                       vector_model=vec_m,
                                       tfidf_model=tfidf_m)
            y_pred_new = model.predict(x_new)
            st.write(f"Comment: {content}")
            display_emotion(y_pred_new)


def fn_restaurant_explore():
    st.write("### :blue[Please select restaurant you would like to analyse: 👇]")
    # restaurant_name = st.text_area(label="Input restaurant name:")

    # Dropdown for selecting restaurant
    name_list = res_df['Restaurant'].unique()
    selected_restaurant = st.selectbox(label='',
                                       options=sorted(name_list),
                                       index=None,
                                       placeholder='Please select Restaurant Name')

    # Display information about the selected restaurant
    restaurant_info = res_df[res_df["Restaurant"] == selected_restaurant]

    if not restaurant_info.empty:
        id_res = restaurant_info['ID'].values[0]
        restaurant_sentiment_analysis(final_df=final_df,
                                      res_df=res_df,
                                      cleanser=cleanser,
                                      id_res=id_res,
                                      top_words=5)


def fn_about_project():

    # Title with colorful background
    st.title(":orange[🚀 Project Introduction]")
    st.markdown("---")

    # Objectives with colorful header
    st.header(":blue[🎯 Objectives:]")
    st.write("""
    Our project aims to analyze sentiment in comments and reviews, with a focus on those related to restaurants. 
    
    We seek to develop a machine learning model capable of accurately predicting sentiment based on textual data. 
    
    Additionally, we aim to provide insights into restaurant comment data through statistical analysis and visualization.
    """)

    # Algorithms Used with colorful header
    st.header(":blue[💡 Algorithms Used:]")
    st.write("""
    For sentiment analysis, we employ the Support Vector Machine (SVM) algorithm. 
    
    SVM is a powerful machine learning technique known for its effectiveness in classification tasks.
    """)

    # Model Building Process with colorful header
    st.header(":blue[🛠️ Model Building Process:]")
    st.write("""
    Our model building process consists of the following steps:

    1. **Data Collection:** Gathering comments and reviews related to restaurants from various sources.
    2. **Data Preprocessing:** Cleaning and preparing the text data for analysis, including removing stopwords and punctuation, and performing tokenization.
    3. **Model Building:** Training the Support Vector Machine (SVM) model on the preprocessed text data.
    4. **Model Evaluation:** Assessing the performance of the trained model using evaluation metrics such as accuracy, precision, recall, and F1-score. Selecting the best-performing model.
    5. **Model Deployment:** Saving the trained model for future use and deployment.
    6. **Real-world Application:** Applying the trained model to analyze sentiment in real-world restaurant comments and reviews.
    7. **Evaluation:** Continuously evaluating and refining the model's performance based on feedback and new data.
    8. **Interface Development:** Creating a user-friendly interface for interacting with the model and viewing analysis results.
    """)

    # Implemented by with colorful header
    st.header(":blue[👨‍💻 Implemented by:]")
    st.write("""
    - Pham Quoc Thai
    - Le Nguyen Duc Tri
    """)


def main():

    with st.sidebar:
        choice = option_menu(menu_title="Main Menu",
                             options=["About the project", "Business Objective", "Emotion Prediction", 'Restaurant Explore'],
                             icons=['info-square', 'lightbulb', 'search', "list-task"])
        st.subheader(f"{choice} Selected")
    if choice is not None:
        if choice == 'About the project':
            st.title(":rainbow[Project 1: Sentiment Analysis]")
            fn_about_project()
        elif choice == 'Business Objective':
            st.title(f":orange[{choice}]")
            fn_business_objective()

        elif choice == 'Emotion Prediction':
            st.title(f":orange[{choice}]")
            fn_new_prediction()
        elif choice == 'Restaurant Explore':
            st.title(f":orange[{choice}]")
            st.write("""
            - Based on data collected from Shopee Food, we have developed a system to assist in analyzing customer segmentation for a specific restaurant.
            
            - Our objective is to generate WordClouds for positive, negative, and neutral comments.
            
            - Through this process, we can identify the top five most frequently occurring words for each respective sentiment category.
            
            - Additionally, we will examine the temporal distribution of comments across different sentiment categories.
            """)
            fn_restaurant_explore()


if __name__ == "__main__":
    main()
