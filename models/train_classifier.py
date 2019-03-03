import sys
import pickle
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def load_data(database_filepath):
    """Loads the data from a database into features and targets.

    Arguments:
        database_filepath - the database file.
    """
    # Load data from the database.
    engine = create_engine(f"sqlite:///{database_filepath}")
    conn = engine.connect()
    df = pd.read_sql_table('messages', con=conn)
    
    # Split into features and targets.
    X = df.message.values
    y = df.as_matrix(columns=df.iloc[:, 4:].columns)
    categories = np.unique(y)
    
    return X, y, categories


def tokenize(text):
    """Splits a sentence into tokens.

    Arguments:
        text - the sentence.
    """
    # Convert to lower case.
    text = text.lower()

    # Remove non-alphanumeric characters.
    text = re.sub(r'[^a-z0-9]', ' ', text)

    # Split into tokens.
    tokens = text.split()
    
    return tokens

def build_model():
    """Builds the classifier.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluates the classifier's performance on the test data.

    Arguments:
        model -- the classifier.
        X_test -- the test features.
        Y_test -- the test targets.
        category_names -- the target classes.
    """
    # Get predictions.
    y_pred = model.predict(X_test)

    # Check results for each output class.
    for i, _ in enumerate(category_names):
        print(classification_report(Y_test[i], Y_test[i]))


def save_model(model, model_filepath):
    """Saves the classifier to a binary file.

    Arguments:
        model -- the classifier.
        model_filepath -- the file to save the classifier to.
    """
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    """Runs the training script.
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