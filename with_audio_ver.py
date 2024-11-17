import speech_recognition as sr 
import pandas as pd 
import numpy as np 
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px 
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
import nltk
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
nltk.download('punkt')

#Page Config
st.set_page_config(
    page_title="SpamVis",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#Datasets
hotel_df = pd.read_csv("hotel_df.csv")
restaurant_df = pd.read_csv("restaurant_df.csv")

#Voice recognition part 
def show_listening_indicator(source): 
    recognizer = sr.Recognizer()
    with st.spinner("Listening..."): 
        audio = recognizer.listen(source)
    return audio 

def get_user_voice_input(): 
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        audio = show_listening_indicator(source)
    try:
        text = recognizer.recognize_google(audio)
        print("You said: " + text)
        return text.lower()  # Convert to lowercase for easier comparison
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return None

#Sidebar
st.title("SpamVis: Multimodal Visual Interactive System for Spam Review Detection")
selected_model = st.sidebar.selectbox("Select model: ", ["BERT","RoBERTa","KNN", "SVM","Decision Tree","Logistics Regression"])
selected_train_dataset = st.sidebar.selectbox("Select train dataset: ",["Restaurant data","Hotel data"])
selected_test_dataset = st.sidebar.selectbox("Select test dataset: ",["Restaurant data", "Hotel data"])

spoken_text = None 

# Add a button to trigger voice input
if st.sidebar.button("Speak your choice"):
  spoken_text = get_user_voice_input()
  print("Spoken Text:", spoken_text)

  if spoken_text:
  # Split the text into keywords
    words = spoken_text.split()

    # Loop through keywords to identify user choices
    for i in range(len(words)):
        if words[i] in ["BERT", "RoBERTa", "KNN", "SVM", "Decision Tree", "Logistics Regression"]:
            selected_model = words[i] 

        if words[i] == "train" and words[i+1] == "on":
            selected_train_dataset = words[i+2]  # Assuming train data follows "train on"
        elif words[i] == "test" and words[i+1] == "on":
            selected_test_dataset = words[i+3]  # Assuming test data follows "test on"


#Generate Results button
if st.sidebar.button("Generate Results") or spoken_text:
    #Create 2 tabs for model results & spam analyzer 
    tab1, tab2, tab3 = st.tabs(["Model results","Review Analyzer","Model Comparision"])
    with tab1: 
    #Main space
        col1, col2 = st.columns(2)
        
        #Define the right df based on user choice
        if selected_model == "BERT": 
                if selected_train_dataset == "Restaurant data": 
                    df = restaurant_df   #bert_restaurant
                    header_text = "Restaurant"

                elif selected_train_dataset == "Hotel data": 
                    df = hotel_df   #bert_hotel 
                    header_text = "Hotel"
                
                else: 
                    header_text = "Unknown"

        elif selected_model == "RoBERTa": 
                if selected_train_dataset == "Restaurant data": 
                    df = restaurant_df  #roberta_restaurant
                    header_text = "Restaurant"

                elif selected_train_dataset == "Hotel data": 
                    df = hotel_df  #roberta_hotel 
                    header_text = "Hotel"
                
                else: 
                    header_text = "Unknown"


        with col1:
            #GRAPH 1: FULL TABLE OF SPAM REVIEWS
            st.header(f"Total Spam Reviews of {header_text} data")

            #TO EDIT: Extract spam tables             
            spam_table = df[df['Label'] == 'Y']

            #Compute percentage of spam reviews detected 
            num_total_reviews = len(df)
            num_spam_reviews = len(spam_table)
            spam_percentage = (num_spam_reviews / num_total_reviews) * 100
            st.write(f"{spam_percentage:.2f}% of the dataset are spam reviews.")
            st.dataframe(spam_table, height=400)

            
            #GRAPH 3: SENTIMENTS
            #Generate random data - will replace later 
            
            np.random.seed(42) 
            df['Sentiment'] = np.random.choice(['Positive', 'Negative'], size=len(restaurant_df))
            
            #Extract all sentiments of spam and non-spam 
            spam_sentiments = df[df['Label'] == 'Y']['Sentiment'].value_counts(normalize=True)
            genuine_sentiments = df[df['Label'] == 'N']['Sentiment'].value_counts(normalize=True)
            
            dataset_name = header_text 
            

            #Sentiment DataFrame
            df_sentiment = pd.DataFrame({
                'Sentiment': spam_sentiments.index,
                'Spam Review': spam_sentiments.values,
                'Genuine Review': genuine_sentiments.values
            })

            # Map sentiment labels to colors
            color_map = {
                'Positive': '#98FB98',  # light green
                'Negative': '#FFB6C1'   # light red
            }

            st.header("Average sentiments of reviews")

            #Combine Spam + Genuine Review Sentiments into 1 DataFrame
            df_combined_sentiment = pd.DataFrame({
                'Sentiment': df_sentiment['Sentiment'],
                'Spam Review': df_sentiment['Spam Review'],
                'Genuine Review': df_sentiment['Genuine Review']
            })

            #Figure with subplots for Spam + Genuine Review Sentiments
            fig_combined_sentiment = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]], subplot_titles=("Spam Review", "Genuine Review"))

            #Spam Pie charts
            fig_combined_sentiment.add_trace(go.Pie(labels=df_combined_sentiment['Sentiment'], 
                                                    values=df_combined_sentiment['Spam Review'],
                                                    marker=dict(colors=[color_map[sentiment] for sentiment in df_combined_sentiment['Sentiment']]),
                                                    hoverinfo='label+percent',
                                                    textinfo='percent+label',
                                                    textposition='inside',
                                                    textfont_color='black',
                                                    hole=0.3),
                                            1, 1)

            #Genuine Pie charts
            fig_combined_sentiment.add_trace(go.Pie(labels=df_combined_sentiment['Sentiment'], 
                                                    values=df_combined_sentiment['Genuine Review'],
                                                    marker=dict(colors=[color_map[sentiment] for sentiment in df_combined_sentiment['Sentiment']]),
                                                    hoverinfo='label+percent',
                                                    textinfo='percent+label',
                                                    textposition='inside',
                                                    textfont_color='black',
                                                    hole=0.3),
                                            1, 2)

            #Update title + size
            fig_combined_sentiment.update_layout(
                                                showlegend=False,
                                                height=400,
                                                width=1800,
                                                font_color = 'black')

            #Show combined figure with two pie charts
            st.plotly_chart(fig_combined_sentiment, use_container_width=True)



        with col2:
            # GRAPH 2: AVG LENGTH BETWEEN SPAM & GENUINE 
            spam_reviews = df[df['Label'] == 'Y']['Review']
            genuine_reviews = df[df['Label'] == 'N']['Review']
            
            avg_words_spam = np.mean([len(review.split()) for review in spam_reviews])
            avg_words_genuine = np.mean([len(review.split()) for review in genuine_reviews])
            avg_words_spam_rounded = round(avg_words_spam)
            avg_words_genuine_rounded = round(avg_words_genuine)

            # Create length comparison df 
            
            df_words_comparison = pd.DataFrame({
                'Average Words': [avg_words_spam_rounded, avg_words_genuine_rounded]
            })

            st.header("Average length of reviews")
            fig_words = px.bar(df_words_comparison, 
                                x='Average Words', 
                                y=['Spam Review', 'Genuine Review'],  # Updated y labels directly in the plotly figure
                                text='Average Words', 
                                orientation='h')
            fig_words.update_traces(marker_color=['#FFA07A', '#ADD8E6'], textfont_color='black', textposition='inside', textfont_size=12)  # Change bar labels color to black and enlarge text size, place text inside bars
            fig_words.update_layout(height=450, yaxis_title=None, xaxis_title=None, yaxis=dict(tickfont=dict(size=14, color='black')), xaxis=dict(tickfont=dict(size=12, color='black')), margin=dict(t=0, b=0)) 
            st.plotly_chart(fig_words, use_container_width=True)

            # GRAPH 4: Display bar chart to compare results between train and test
            st.header("Model's Training & Testing Results")
            data = {
                'Model': ['BERT', 'BERT', 'RoBERTa', 'KNN', 'LR'],
                'Data': ['Hotel data', 'Restaurant data', 'Hotel data', 'Restaurant data', 'Restaurant data'],
                'Accuracy': [0.85, 0.78, 0.93, 0.92, 0.88],
                'Recall': [0.87, 0.81, 0.94, 0.75, 0.89],
                'Precision': [0.84, 0.77, 0.91, 0.72, 0.87],
                'Auc': [0.90, 0.82, 0.95, 0.76, 0.88]
            }
            model_results = pd.DataFrame(data)

            filtered_train_results = model_results[(model_results['Model'] == 'BERT') & (model_results['Data'] == 'Restaurant data')]
            filtered_test_results = model_results[(model_results['Model'] == 'BERT') & (model_results['Data'] == 'Hotel data')]
            merged_results = pd.concat([filtered_train_results, filtered_test_results])

            # Define colors
            colors = ['#FFA07A', '#98FB98', '#ADD8E6', '#FFB6C1']

            # Create the bar chart
            fig, ax = plt.subplots(figsize=(18, 10))
            merged_results.plot(kind='bar', ax=ax, legend=True, width=0.7, color=colors)

            # Customize the plot
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)  # Enlarged legend series
            ax.set_xticklabels(['Training', 'Testing'], rotation=0, fontsize=24)
            ax.set_yticklabels([f'{x:.0%}' for x in ax.get_yticks()], fontsize=20)  # Enlarged y-axis tick labels
            ax.set_ylim(0, 1)

            # Remove the outer border line
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Display data values on each bar as percentages (enlarged)
            for p in ax.patches:
                ax.annotate(f'{p.get_height()*100:.0f}%', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                            va='center',
                            xytext=(0, 20), textcoords='offset points', fontsize=16)  # Enlarged data values

            # Display the plot in Streamlit
            st.pyplot(fig)

            
    with tab2: 
        st.subheader("Analyze Review Text")
       