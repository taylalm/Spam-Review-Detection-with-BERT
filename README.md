# SpamVis - A Visual Interactive System for Spam Review Detection 

SpamVis is an interactive tool that combines several machine learning and deep learning models with real-time data visualization to tackle the growing issue of spam reviews. SpamVis is designed to enable users to detect and analyze fake reviews across industries like e-commerce, food & beverage, hospitality, etc. 

## Features

1. Advanced Spam Detection
- Utilizes machine learning models: Support Vector Machines (SVM), Logistic Regression (LR), Decision Trees (DT), and deep learning models: BERT and RoBERTa.
- Provides accuracy, precision, recall, and AUC metrics for evaluation.

2. Interactive Dashboard
- Offers customizable offline analysis with support for restaurant and hotel datasets.
- Real-time analysis of reviews from social media platforms like Facebook, Instagram, and YouTube.
- Visualizations include bar graphs, sentiment analysis, and comparison plots.

3. Cross-Domain and Cross-Platform Analysis
- Analyze reviews across industries and platforms.
- Offers insights into domain-specific spam behaviors and sentiment manipulation.

4. Custom Dataset Support
- Users can upload their own datasets for personalized analysis.

## Dashboard Functionalities 

1. Offline Analysis
- Choose between preloaded hotel and restaurant datasets or upload your own dataset in .csv format.
- Select the model to apply (e.g., BERT, RoBERTa, SVM).
- View results through visual metrics, such as: Spam vs. non-spam word length distribution; Sentiment distribution among spam and genuine reviews; Accuracy, precision, and recall for the selected models.

2. Real-Time Analysis
- Select a platform (e.g., Facebook, YouTube).
- Fetch and analyze reviews in real-time using trained ML/DL models.
- View spam detection results and sentiment trends instantly.
Datasets

## Preloaded datasets include:
- YelpCHI Restaurant Dataset: ~60,000 reviews.
- YelpCHI Hotel Dataset: ~5,000 reviews.
- Metadata includes product IDs, reviewer IDs, timestamps, and spam annotations.

## Models 
1. Machine Learning Models:
- Support Vector Machines (SVM)
- Logistic Regression (LR)
- Decision Trees (DT)
2. Deep Learning Models:
- BERT (Bidirectional Encoder Representations from Transformers)
- RoBERTa
3. Visualization:
- Real-time dashboards with pandas, seaborn and Plotly.
- SerpAPI for fetching real-time online reviews.

## Results

- BERT achieved the highest accuracy of 86.03% on the restaurant dataset.
- RoBERTa closely followed with competitive performance in precision and recall.
- SVM outperformed other traditional ML models but fell behind DL approaches.

## Future Improvements

- Expansion to additional domains such as healthcare and automotive reviews.
- Integration of multimodal interactions like voice and gestures.
- Enhanced real-time functionality with chatbot support for user-friendly analysis.
