# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet'])

import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle

import sqlite3
from sqlalchemy.engine.url import URL

def load_data(database_filepath):
    """Loads database and reads in messages as well as category dummies from etlOutput table
    
    Args: 
        database_filepath (string): user-provided path to database
		
    Returns: 
        None
    """
    
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql('etlOutput', engine).dropna()
    
    X = df.message.values
    Y = df.iloc[:,4:40].values
    
    # creating a list of dataframe columns 
    category_names = list(df) 
    
    return X, Y, category_names
    pass

def tokenize(text):
    """Breaks a text into individual tokens, lemmatizes them, lowers case
    
    Args: 
        text (string): a snippet of text
		
    Returns: 
        tokens (array): list of words and any other tokens
    """

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    pass

def build_model():
    """Prepares ML pipeline
    
    Args: 
        None
		
    Returns: 
        model (object): scikitlearn GridSearchCV 
    """

    forest = RandomForestClassifier(n_estimators=100, random_state=1)

    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(forest, n_jobs=-1))
        ])
    
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
#         'vect__max_df': (0.5, 0.75, 1.0),
#         'vect__max_features': (None, 5000, 10000),
#         'tfidf__use_idf': (True, False),
#         'n_estimators': [50, 100, 200],
#         'min_samples_split': [2, 3, 4],
    }

    model = GridSearchCV(pipeline, param_grid=parameters)
    return model
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    """Uses model to predict message categories then evaluates predictions against actual outcomes
    
    Args: 
        model (object): scikitlearn GridSearchCV
        X_test (array): array of message texts
        Y_test (array): matrix of dummies indicating what categories were relevant to a given message (row, in this case)
        category_names (array): list of category names
		
    Returns: 
        None
    """
    
    Y_pred = model.predict(X_test)
    for i in range(4,34):
        print('***' + str(category_names[i]) + '***')
        print(classification_report(Y_test[:,i],Y_pred[:,i]))
    pass

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))
    pass

def main():
    """Trains and evaluates a model using message/category data, then saves that model in pickle file
    
    Args: 
        None
		
    Returns: 
        None
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