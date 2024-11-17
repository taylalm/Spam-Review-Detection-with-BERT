from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier



def preprocess_data(df):
    # Convert date column to numeric format
    df['Date'] = pd.to_datetime(df['Date']).astype(int)
    
    return df

def train_knn_model(train_df, test_df):
    # Preprocess the data
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)
    
    # Extract features and target for training
    train_features = train_df[['Date', 'Length', 'MeaningWords']]
    train_target = train_df['Label']
    
    # Extracting features and target for testing
    test_features = test_df[['Date', 'Length', 'MeaningWords']]
    test_target = test_df['Label']
    
    # Initialize and train the KNN model
    knn_model = KNeighborsClassifier(n_neighbors=5) 
    knn_model.fit(train_features, train_target)
    
    # Make predictions on the test set
    predictions = knn_model.predict(test_features)
    
    # Calculate performance metrics for training set
    train_predictions = knn_model.predict(train_features)
    train_accuracy = accuracy_score(train_target, train_predictions)
    train_precision = precision_score(train_target, train_predictions, average='weighted')
    train_recall = recall_score(train_target, train_predictions, average='weighted')
    train_f1 = f1_score(train_target, train_predictions, average='weighted')
    
    # Calculate performance metrics for test set
    test_accuracy = accuracy_score(test_target, predictions)
    test_precision = precision_score(test_target, predictions, average='weighted')
    test_recall = recall_score(test_target, predictions, average='weighted')
    test_f1 = f1_score(test_target, predictions, average='weighted')
    
    return {
        'Train Accuracy': train_accuracy,
        'Train Precision': train_precision,
        'Train Recall': train_recall,
        'Train F1-score': train_f1,
        'Test Accuracy': test_accuracy,
        'Test Precision': test_precision,
        'Test Recall': test_recall,
        'Test F1-score': test_f1
    }

def train_svm_model(train_df, test_df):
    # Preprocess the data if needed
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)
    
    # Extract features and target for training
    train_features = train_df[['Date', 'Length', 'MeaningWords']]
    train_target = train_df['Label']
    
    # Extracting features and target for testing
    test_features = test_df[['Date', 'Length', 'MeaningWords']]
    test_target = test_df['Label']
    
    # Initialize and train the SVM model
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(train_features, train_target)
    
    # Make predictions on the test set
    predictions = svm_model.predict(test_features)
    
    # Calculate performance metrics for training set
    train_predictions = svm_model.predict(train_features)
    train_accuracy = accuracy_score(train_target, train_predictions)
    train_precision = precision_score(train_target, train_predictions, average='weighted')
    train_recall = recall_score(train_target, train_predictions, average='weighted')
    train_f1 = f1_score(train_target, train_predictions, average='weighted')
    
    # Calculate performance metrics for test set
    test_accuracy = accuracy_score(test_target, predictions)
    test_precision = precision_score(test_target, predictions, average='weighted')
    test_recall = recall_score(test_target, predictions, average='weighted')
    test_f1 = f1_score(test_target, predictions, average='weighted')
    
    return {
        'Train Accuracy': train_accuracy,
        'Train Precision': train_precision,
        'Train Recall': train_recall,
        'Train F1-score': train_f1,
        'Test Accuracy': test_accuracy,
        'Test Precision': test_precision,
        'Test Recall': test_recall,
        'Test F1-score': test_f1
    }

def train_gradient_boosting_model(train_df, test_df):
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)

    train_features = train_df[['Date', 'Length', 'MeaningWords']]
    train_target = train_df['Label']

    test_features = test_df[['Date', 'Length', 'MeaningWords']]
    test_target = test_df['Label']

    gb_model = GradientBoostingClassifier(random_state=42)
    gb_model.fit(train_features, train_target)

    predictions = gb_model.predict(test_features)

    train_predictions = gb_model.predict(train_features)
    train_accuracy = accuracy_score(train_target, train_predictions)
    train_precision = precision_score(train_target, train_predictions, average='weighted')
    train_recall = recall_score(train_target, train_predictions, average='weighted')
    train_f1 = f1_score(train_target, train_predictions, average='weighted')

    test_accuracy = accuracy_score(test_target, predictions)
    test_precision = precision_score(test_target, predictions, average='weighted')
    test_recall = recall_score(test_target, predictions, average='weighted')
    test_f1 = f1_score(test_target, predictions, average='weighted')

    return {
        'Train Accuracy': train_accuracy,
        'Train Precision': train_precision,
        'Train Recall': train_recall,
        'Train F1-score': train_f1,
        'Test Accuracy': test_accuracy,
        'Test Precision': test_precision,
        'Test Recall': test_recall,
        'Test F1-score': test_f1
    }

def train_logistic_regression_model(train_df, test_df):
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)

    train_features = train_df[['Date', 'Length', 'MeaningWords']]
    train_target = train_df['Label']

    test_features = test_df[['Date', 'Length', 'MeaningWords']]
    test_target = test_df['Label']

    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(train_features, train_target)

    predictions = lr_model.predict(test_features)

    train_predictions = lr_model.predict(train_features)
    train_accuracy = accuracy_score(train_target, train_predictions)
    train_precision = precision_score(train_target, train_predictions, average='weighted')
    train_recall = recall_score(train_target, train_predictions, average='weighted')
    train_f1 = f1_score(train_target, train_predictions, average='weighted')

    test_accuracy = accuracy_score(test_target, predictions)
    test_precision = precision_score(test_target, predictions, average='weighted')
    test_recall = recall_score(test_target, predictions, average='weighted')
    test_f1 = f1_score(test_target, predictions, average='weighted')

    return {
        'Train Accuracy': train_accuracy,
        'Train Precision': train_precision,
        'Train Recall': train_recall,
        'Train F1-score': train_f1,
        'Test Accuracy': test_accuracy,
        'Test Precision': test_precision,
        'Test Recall': test_recall,
        'Test F1-score': test_f1
    }

def train_random_forest_model(train_df, test_df):
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)

    train_features = train_df[['Date', 'Length', 'MeaningWords']]
    train_target = train_df['Label']

    test_features = test_df[['Date', 'Length', 'MeaningWords']]
    test_target = test_df['Label']

    rf_model = RandomForestClassifier(max_depth = 5, n_estimators=100, random_state=42)
    rf_model.fit(train_features, train_target)

    predictions = rf_model.predict(test_features)

    train_predictions = rf_model.predict(train_features)
    train_accuracy = accuracy_score(train_target, train_predictions)
    train_precision = precision_score(train_target, train_predictions, average='weighted')
    train_recall = recall_score(train_target, train_predictions, average='weighted')
    train_f1 = f1_score(train_target, train_predictions, average='weighted')

    test_accuracy = accuracy_score(test_target, predictions)
    test_precision = precision_score(test_target, predictions, average='weighted')
    test_recall = recall_score(test_target, predictions, average='weighted')
    test_f1 = f1_score(test_target, predictions, average='weighted')

    return {
        'Train Accuracy': train_accuracy,
        'Train Precision': train_precision,
        'Train Recall': train_recall,
        'Train F1-score': train_f1,
        'Test Accuracy': test_accuracy,
        'Test Precision': test_precision,
        'Test Recall': test_recall,
        'Test F1-score': test_f1
    }

def train_adaboost_model(train_df, test_df):
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)
    
    train_features = train_df[['Date', 'Length', 'MeaningWords']]
    train_target = train_df['Label']
    
    test_features = test_df[['Date', 'Length', 'MeaningWords']]
    test_target = test_df['Label']
    
    adaboost_model = AdaBoostClassifier(n_estimators=50, random_state=42)
    adaboost_model.fit(train_features, train_target)
    
    predictions = adaboost_model.predict(test_features)
    
    train_predictions = adaboost_model.predict(train_features)
    train_accuracy = accuracy_score(train_target, train_predictions)
    train_precision = precision_score(train_target, train_predictions, average='weighted')
    train_recall = recall_score(train_target, train_predictions, average='weighted')
    train_f1 = f1_score(train_target, train_predictions, average='weighted')
    
    test_accuracy = accuracy_score(test_target, predictions)
    test_precision = precision_score(test_target, predictions, average='weighted')
    test_recall = recall_score(test_target, predictions, average='weighted')
    test_f1 = f1_score(test_target, predictions, average='weighted')
    
    return {
        'Train Accuracy': train_accuracy,
        'Train Precision': train_precision,
        'Train Recall': train_recall,
        'Train F1-score': train_f1,
        'Test Accuracy': test_accuracy,
        'Test Precision': test_precision,
        'Test Recall': test_recall,
        'Test F1-score': test_f1
    }
