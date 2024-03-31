import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#from sklearn.model_selection import train_test_split  
#from sklearn.metrics import accuracy_score 
#from sklearn.metrics import confusion_matrix
#from sklearn. metrics import classification_report, roc_auc_score, roc_curve
import pickle
#import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
#from sklearn import metrics
import seaborn as sns
import regex
from Viet_lib.Viet_lib import *
   
def preprocessing_text(text):
    
    # convert to lower_case
    clean_text = text.lower()
    
    # remove "'"
    clean_text = regex.sub("'", '', clean_text)
    
    # convert multiple dot to 1 dot    
    clean_text = regex.sub(r'\.+', ".", clean_text)
    
    # Preprocessing for punctuations, uppercases, and some wrong words, and emoji
    clean_text = process_text(str(clean_text), emoji_dict, teen_dict, wrong_lst)
    
    clean_text = convert_unicode(str(clean_text))
    
    clean_text = process_postag_thesea(str(clean_text))
    
    clean_text = remove_stopword(str(clean_text), stopwords_lst)
    
    #initialize countvectorizer
    vectorizer = pickle.load(open('resources/countvectorizer_model.pkl', 'rb'))
    
    #intialize TFIDF
    tfidf = pickle.load(open('resources/tfidf_model.pkl', 'rb'))
    
    #transform X_test
    _pre_X_test = vectorizer.transform([clean_text])
    _pre_X_test = tfidf.transform(_pre_X_test)
    
    return _pre_X_test
    
def fn_business_objective():  
    #st.subheader("Business Objective")
    
    # Upload file
    uploaded_file = st.file_uploader("Choose a file", type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, encoding='latin-1')
        data.to_csv("resources/new_file.csv", index = False)
    
    st.subheader("EDA tập dữ liệu")
    st.write("""Dữ liệu: Gồm 2 tập dữ liệu về nhà hàng và bình luận khách hàng""")
    st.subheader("1. Dữ liệu bình luận")
    some_data = pd.read_csv('resources/some_data.csv')
    st.write("""
    #### 1.1. Some data""")
    st.dataframe(some_data) 
    st.write("""
    * Có tất cả 29959 bình luận""")
    
    st.write("""
    #### 1.2. Biểu đồ phân bố rating""")
    st.image('resources/rating_plot.png')
    st.write("""
    * Rating score có giá trị từ 0 đến 10
    * Rating phân bố nhiều từ 7 trở lên""")
    
    st.write("""
    #### 1.3. Số lượng loại bình luận: Not like - Neutral - Like""") 
    st.image('resources/label_count.png')
    st.write("""
    * Bình luận Like chiếm đa số""")
    
    st.write("""
    #### 1.4. Một số từ xuất hiện nhiều trong bình luận Not like""") 
    st.image('resources/not_like_comment.png')
    
    st.write("""
    #### 1.5. Một số từ xuất hiện nhiều trong bình luận Neutral""") 
    st.image('resources/neutral_comment.png')
    
    st.write("""
    #### 1.6. Một số từ xuất hiện nhiều trong bình luận Like""") 
    st.image('resources/like_comment.png')
        
    st.subheader("2. Dữ liệu nhà hàng")
    some_restaurant = pd.read_csv('resources/some_restaurant.csv')
    st.write("""
    #### 2.1. Some data""")
    st.dataframe(some_restaurant) 
    st.write("""
    * Có tất cả 1622 nhà hàng""")
    
    st.write("""
    #### 2.2. Biểu đồ phân bố nhà hàng ở các quận""")
    st.image('resources/restaurant_districts.png')
    
    st.write("""
    #### 2.3. Mức giá trung bình tại các quận""") 
    st.image('resources/price_district.png')
    st.write("""
    * Mức giá trung bình tại quận 1, 2, 5 cao hơn các quận khác.
    * Mức giá trung bình thấp nhất ở quận 10, 11, 12.""")
    
def display_emotion(emotion_index):
    if emotion_index == 0:
        st.image("resources/not_like_icon.png", caption="Not Like", width=200)
    elif emotion_index == 1:
        st.image("resources/neutral_icon.png", caption="Neutral", width=200)
    elif emotion_index == 2:
        st.image("resources/like_icon.png", caption="Like", width=200)
        
def fn_new_prediction():
    #st.subheader("New Prediction")
    # load model classication
    pkl_filename = "resources/project_1_model_SVC.sav"  
    model = pickle.load(open(pkl_filename, 'rb'))
    st.subheader("Select data")
    flag = False
    lines = None
    type = st.radio("Upload data or Input data?", options=("Input", "Upload"))
    st.write("""
    ##### Example:""")
    st.write('Like: Quán này ngon lắm đó mọi người!!!')
    st.write('Neutral: Quán này tạm ổn thôi à.')
    st.write('Not like: Mọi người nên tránh xa quán này nha, dở lắm.')
    if type=="Upload":
        # Upload file
        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1, header=None)
            st.dataframe(lines)
            # st.write(lines.columns)
            lines = lines[0]     
            flag = True       
    if type=="Input":        
        content = st.text_area(label="Input your comment:")
        
        
        if content!="":
            #lines = np.array([content])
            
            x_new = preprocessing_text(content)  
            #st.write(x_new) 
                         
            y_pred_new = model.predict(x_new)       
            #st.code("New predictions (0: Not Like, 1: Neutral, 2: Like): " + str(y_pred_new))
            st.write(f"Comment: {content}")
            display_emotion(y_pred_new)
            
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=600, background_color ='white', max_words=100).generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    
def fn_restaurant_explore():
    st.subheader("Restaurant Explore")  
    data = pd.read_csv('resources/restaurant_comment.csv')
    
    st.write('Chọn tên một nhà hàng để xem một số thông tin của nhà hàng đó.')
    #restaurant_name = st.text_area(label="Input restaurant name:")
    
    # Dropdown for selecting restaurant
    name_list = data['Restaurant'].unique()
    selected_restaurant = st.selectbox("Select a restaurant:", sorted(name_list))
    
    # Display information about the selected restaurant
    restaurant_info = data[data["Restaurant"] == selected_restaurant]
    
    if not restaurant_info.empty:
        #Display rating and word cloud
        #Filter the dataset based on the restaurant name provided by the user
        #restaurant_df = data[data['Restaurant'] == restaurant_name]
        #Extract ratings and comments
        ratings = restaurant_info['Rating']
        rating_score = ratings.mean()
        price = restaurant_info['Price'].values[0]
        address = restaurant_info['Address'].values[0]
        
        st.write('Restaurant: '+str(selected_restaurant))
        st.write("Average Rating: "+str(round(rating_score,2))+'/10')
        st.write("Price: "+str(price))
        st.write("Location: "+str(address))
        st.write("""
        #### Một số từ ngữ xuất hiện nhiều trong comment của khách hàng""")
        
        comments = ' '.join(restaurant_info['Comment_new'])
        generate_wordcloud(comments)
        
        
        
    
def fn_about_project():
    st.title("Giới thiệu Project")

    st.header("Mục tiêu:")
    st.write("""
    - Xây dựng mô hình dự đoán cảm xúc của bình luận
    - Thống kê dữ liệu bình luận của nhà hàng
    """)

    st.header("Thuật toán sử dụng:")
    st.write("""
    - Machine Learning: Support Vector Machine
    """)

    st.header("Quy trình xây dựng mô hình:")
    st.write("""
    1. Thu thập dữ liệu
    2. Tiền xử lý dữ liệu tiếng việt
    3. Xây dựng mô hình
    4. Đánh giá mô hình => Chọn mô hình tốt nhất
    5. Lưu mô hình
    6. Áp dụng mô hình vào thực tế
    7. Đánh giá
    8. Xây dựng giao diện và sử dụng
    """)

    st.header("Người thực hiện:")
    st.write("""
    Phạm Quốc Thái
    """)
    
def main():
    st.title("Project 1: Sentiment Analysis")
    
    with st.sidebar:
        choice = option_menu("Main Menu", ["About the project", "Business Objective", "Emotion Prediction", 'Restaurant Explore'],
                      icons=['info-square', 'lightbulb', 'search', "list-task"])    
        st.subheader(f"{choice} Selected")                      
    choice
       
    
    if choice is not None:
            if choice == 'About the project':
                fn_about_project()       
            elif choice == 'Business Objective':
                fn_business_objective()
            
            elif choice == 'Emotion Prediction':
                fn_new_prediction()
            elif choice == 'Restaurant Explore':
                fn_restaurant_explore()
                
if __name__ == "__main__":
    main()


