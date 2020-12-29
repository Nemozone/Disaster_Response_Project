import sys
import pickle
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd

from sqlalchemy import create_engine
import sqlite3

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    The StartingVerbExtractor represents the starting verbs in each given text if exists
        class is inheritted from BaseEstimator and TransformerMixin classes in scikit-learn library
    """

    def starting_verb(self, text):
        """
        Returns a binary checking if the text starts with a verb or not
        
        Parameters:
            text (str): each message of the 'messages' column in df
            
        Returns:
            Bolean: True or False
        """
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        """
        Fits the transformer to the data set
        
        Parameters:
            x (ndarray/pandas DataFrame): Features or inputs from the main data frame
            y (ndarray/pandas DataFrame): Targets from the main data frame
            
        Returns:
            Nothing
        """
        return self

    def transform(self, X):
        """
        Returns transformed data 
        
        Parameters:
            X (ndarray/DataFrame): Features or inputs from the main data frame
        
        Returns:
            X_tagged (DataFrame): transformed data
        """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def load_data(database_filepath):
    """
    Loads data from database and extract inputs, targets, and category names
    
    Parameters:
        database_filepath (str): file path for the restored database file
        
    Returns:
        X (ndarray): inputs extracted from the main dataframe
        Y (ndarray): targets extracted from the main dataframe
        category_names: names of categorues extracted from main dataframe category column names
    """

    # load data from database
    engine = create_engine('sqlite:///'+ str(database_filepath))
    df = pd.read_sql_table('DisasterResponse.db', engine)

    # Assign message column as inputs and category columns as targets
    X = df.message.values
    Y = df.iloc[:,4:].values
    category_names = df.iloc[:,4:].columns
    
    return X,Y,category_names


def tokenize(text):
    """
    Returns clean tokenized text removing all the unnecessary parameters
    
    Parameters:
        text (str): message texts from 'messages' column in the main dataframe
        
    Returns:
        clean_tokens (list): a list of cleaned tokenized text        
    """

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    """
    Returns built and tuned model using pipeline
    
    Parameters:
        No arguments
        
    Returns:
        cv (estimator): tuned model
    """

    pipeline = Pipeline([
        ('Features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(DecisionTreeClassifier()))
    ])
    
    # now we can perform another grid search on this new estimator to be sure we have the best parameters
    parameters = {
        'Features__text_pipeline__vect__max_df': [0.5,1.0],
        'Features__text_pipeline__tfidf__smooth_idf': (True, False)    
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evalute built model and prints a classification report
    
    Parameters:
        model (estimator): built and tuned model from build_model function
        X_test (ndarray): input test splitted data
        Y_test (ndarray): target test splitted data
        category_names (list): category names
        
    Returns:
        prints the classification report consisting accuracy, recall, f1-score, and support
    """
    
    predicted = model.predict(X_test)
    y_test_df = pd.DataFrame(Y_test, columns=category_names)
    predicted_df = pd.DataFrame(predicted, columns = category_names)
    
    print(classification_report(y_test_df, predicted_df, target_names= category_names, zero_division=1))

def save_model(model, model_filepath):
    """
    Saves model in a pickle file
    
    Parameters:
        model (estimator): built and tuned model from build_model function
        model_filepath (str): the filepath in which we want to restore the pickle file
        
    Returns:
        Nothing
    """
    with open(str(model_filepath), 'wb') as file:
        pickle.dump(model, file)


def main():
    """
    Executes all the modeling by running all the functions and saves the 
    model into the adressed pickle file.
    
    Paramenters:
        No arguments
    
    Returns:
        Nothing
    
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()