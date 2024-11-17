import pandas as pd 
import numpy as np 
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px 
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
import nltk
import serpapi
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#nltk.download('punkt')


#Page Config
st.set_page_config(
    page_title="SpamVis",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

# Set the title of the sidebar
st.sidebar.title("Offline Data Analysis")

# Define the path to your zip file
zip_file_path = "restaurant_df.zip"

# Define the path to your zip file
zip_file_path1 = "hotel_df.zip"

#Datasets
#hotel_df = pd.read_csv("hotel_df.csv")
#restaurant_df = pd.read_csv("restaurant_df.csv")

# Open the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # List all files in the zip archive
    file_list = zip_ref.namelist()
    print("Files in the zip archive:", file_list)
    
    # Assuming there's only one CSV file, or you know the name of the CSV file
    csv_file_name = "restaurant_df.csv"  # Replace with the actual file name if different
    
    # Extract and read the CSV file into a DataFrame
    with zip_ref.open(csv_file_name) as csv_file:
        restaurant_df = pd.read_csv(csv_file)

# Open the zip file for hotel data
with zipfile.ZipFile(zip_file_path1, 'r') as zip_ref:
    # List all files in the zip archive
    file_list = zip_ref.namelist()
    print("Files in the zip archive:", file_list)
    
    # Assuming there's only one CSV file, or you know the name of the CSV file
    csv_file_name1 = "hotel_df.csv"  # Replace with the actual file name if different
    
    # Extract and read the CSV file into a DataFrame
    with zip_ref.open(csv_file_name1) as csv_file:
        hotel_df = pd.read_csv(csv_file)

#Sidebar
st.title("SpamVis: Multimodal Visual Interactive System for Spam Review Detection")
selected_model = st.sidebar.selectbox("Select model: ", ["BERT","RoBERTa", "SVM","Decision Tree","Logistics Regression"])
selected_train_dataset = st.sidebar.selectbox("Select train dataset: ",["Restaurant data","Hotel data"])

#Test dataset option
test_option = st.sidebar.radio("Choose Test Data Option:", ("Select Predefined Test Dataset", "Upload Your Own CSV File"))

if test_option == "Select Predefined Test Dataset":
    selected_test_dataset = st.sidebar.selectbox("Select test dataset: ",["Restaurant data", "Hotel data"])
else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        custom_test_df = pd.read_csv(uploaded_file)


        

#Generate Results button
if st.sidebar.button("Generate Results"):
    #Create 2 tabs for model results & spam analyzer 
    tab1, tab2 = st.tabs(["Model results","Review Analyzer"])
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

        elif selected_model == "RoBERTa": 
                if selected_train_dataset == "Restaurant data": 
                    df = restaurant_df  #roberta_restaurant
                    header_text = "Restaurant"

                elif selected_train_dataset == "Hotel data": 
                    df = hotel_df  #roberta_hotel 
                    header_text = "Hotel"

        elif selected_model == "SVM": 
                if selected_train_dataset == "Restaurant data": 
                    df = restaurant_df  #roberta_restaurant
                    header_text = "Restaurant"

                elif selected_train_dataset == "Hotel data": 
                    df = hotel_df  #roberta_hotel 
                    header_text = "Hotel"
        elif selected_model == "Decision Tree": 
                if selected_train_dataset == "Restaurant data": 
                    df = restaurant_df  #roberta_restaurant
                    header_text = "Restaurant"

                elif selected_train_dataset == "Hotel data": 
                    df = hotel_df  #roberta_hotel 
                    header_text = "Hotel"

        elif selected_model == "Logistics Regression": 
                if selected_train_dataset == "Restaurant data": 
                    df = restaurant_df  #roberta_restaurant
                    header_text = "Restaurant"

                elif selected_train_dataset == "Hotel data": 
                    df = hotel_df  #roberta_hotel 
                    header_text = "Hotel"


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
                'Model': ['BERT', 'BERT', 'RoBERTa', 'RoBERTa', "SVM", "SVM", "Decision Tree", "Decision Tree", "Logistics Regression", "Logistics Regression"],
                'Data': ['Hotel data', 'Restaurant data', 'Hotel data', 'Restaurant data',  'Hotel data', 'Restaurant data',  'Hotel data', 'Restaurant data',  'Hotel data', 'Restaurant data'],
                'Accuracy': [0.85, 0.78, 0.93, 0.92, 0.80, 0.77, 0.88, 0.80, 0.85, 0.81],
                'Recall': [0.87, 0.81, 0.94, 0.75, 0.88, 0.83, 0.85, 0.78, 0.78, 0.76],
                'Precision': [0.84, 0.77, 0.91, 0.72, 0.79, 0.74, 0.77, 0.77, 0.80, 0.75],
                'Auc': [0.90, 0.82, 0.95, 0.76, 0.87, 0.81, 0.89, 0.80, 0.88, 0.87]
            }
            model_results = pd.DataFrame(data)

            filtered_train_results = model_results[(model_results['Model'] == 'BERT') & (model_results['Data'] == 'Restaurant data')]
            filtered_test_results = model_results[(model_results['Model'] == 'BERT') & (model_results['Data'] == 'Hotel data')]
            merged_results = pd.concat([filtered_train_results, filtered_test_results])

            filtered_train_results = model_results[(model_results['Model'] == 'RoBERTa') & (model_results['Data'] == 'Restaurant data')]
            filtered_test_results = model_results[(model_results['Model'] == 'RoBERTa') & (model_results['Data'] == 'Hotel data')]
            merged_results = pd.concat([filtered_train_results, filtered_test_results])

            filtered_train_results = model_results[(model_results['Model'] == 'SVM') & (model_results['Data'] == 'Restaurant data')]
            filtered_test_results = model_results[(model_results['Model'] == 'SVM') & (model_results['Data'] == 'Hotel data')]
            merged_results = pd.concat([filtered_train_results, filtered_test_results])

            filtered_train_results = model_results[(model_results['Model'] == 'Decision Tree') & (model_results['Data'] == 'Restaurant data')]
            filtered_test_results = model_results[(model_results['Model'] == 'Decision Tree') & (model_results['Data'] == 'Hotel data')]
            merged_results = pd.concat([filtered_train_results, filtered_test_results])

            filtered_train_results = model_results[(model_results['Model'] == 'Logistics Regression') & (model_results['Data'] == 'Restaurant data')]
            filtered_test_results = model_results[(model_results['Model'] == 'Logistics Regression') & (model_results['Data'] == 'Hotel data')]
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

       

# Set the title of the sidebar
st.sidebar.title("Online Data Analysis")

# Sidebar creation for selecting app and number of reviews
with st.sidebar:
    # App selection
    app_names = {
        'Facebook': 'com.facebook.katana',
        'Instagram': 'com.instagram.android',
        'Messenger': 'com.facebook.orca',
        'WhatsApp': 'com.whatsapp',
        'YouTube': 'com.google.android.youtube'
    }
    selected_app = st.selectbox('Select an app:', list(app_names.keys()))

    # Number of reviews to fetch
    num_reviews = st.slider('Select number of reviews to fetch:', min_value=1, max_value=1000, value=100)

# Fetch Data button
if st.sidebar.button("Fetch Data"):
    import streamlit as st
    import pandas as pd
    import numpy as np
    import serpapi
    from transformers import BertTokenizer, BertForSequenceClassification
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Function to load the BERT model and tokenizer
    @st.cache(allow_output_mutation=True)
    def get_model():
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained("pnichite/YTFineTuneBert")
        return tokenizer, model

    # Function to predict the toxicity of a given text using the BERT model
    def predict_toxicity(tokenizer, model, text):
        test_sample = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        output = model(**test_sample)
        y_pred = np.argmax(output.logits.detach().numpy(), axis=1)
        return y_pred[0]

    # Function to fetch data using SerpApi and save it as a CSV
    def fetch_and_display_data(product_id, num_reviews):
        client = serpapi.Client(api_key="20b3d5bdd2922f1cb83efa7825287649880b87d556e7186d736770506e10566e")
        results = client.search(
            engine="google_play_product",
            product_id=product_id,
            store="apps",
            all_reviews="true",
            num=num_reviews
        )
        data = results['reviews']
        df = pd.DataFrame(data)
        return df

    # Function to classify reviews as positive or negative based on toxicity predictions
    def classify_reviews(df, tokenizer, model):
        df['Prediction'] = df['snippet'].apply(lambda x: predict_toxicity(tokenizer, model, x))
        df['Review Type'] = np.where(df['Prediction'] == 1, 'Real', 'Spam')
        df['Sentiment'] = np.where(df['Prediction'] == 1, 'Positive', 'Negative')
        return df

    # Streamlit web app
    st.title('App Reviews Analyzer')

    # Get the BERT model and tokenizer
    tokenizer, model = get_model()

    # Create two columns layout
    col1, col2 = st.columns(2)

    # Column 1
    with col1:
        # Fetch data and display DataFrame when button is clicked
        st.write('Fetching data...')
        df = fetch_and_display_data(app_names[selected_app], num_reviews)
        st.write('Data fetched successfully!')
        st.write('Fetched DataFrame:')
        st.dataframe(df)

        # Classify reviews
        st.write('Classifying reviews...')
        df = classify_reviews(df, tokenizer, model)
        st.write('Reviews classified successfully!')
        st.write('Classified DataFrame:')
        st.dataframe(df)

    # Column 2
    with col2:
        # Sample data
        data = {
            'Review Type': ['Spam', 'Real', 'Positive', 'Negative'],
            'Percentage': [5.25, 94.75, 62.50, 37.50]  # Example percentages
        }
        percentages_df = pd.DataFrame(data)

        # Create a bar plot
        fig = go.Figure(go.Bar(
            x=percentages_df['Percentage'],
            y=percentages_df['Review Type'],
            orientation='h'
        ))

        # Update layout
        fig.update_layout(
            title="Percentage Summary",
            xaxis_title="Percentage",
            yaxis_title="Review Type"
        )

        # Display the plot in column 2
        st.write('Percentages Summary:')
        st.plotly_chart(fig, use_container_width=True)
