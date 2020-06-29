import re
import sys
import nltk
import sqlalchemy
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from joblib import dump, load
from tokenize_messages import tokenize
#from workspace_utils import active_session

def load_data(database_filepath):
    '''loads data from database_filepath
    INPUT - database_filepath (str)
    OUTPUT - X (df) dataframe with messages
            Y(df) dataframe with categories
            category_names (array) Array of category names'''
    engine = sqlalchemy.create_engine(f"sqlite:///{database_filepath}")
    data = pd.read_sql_table('disaster_response', engine)
    X = data.message
    Y = data.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = Y.columns
    return X, Y, category_names

def build_model():
    '''performs a grid search on  GradientBoostingClassifier 
    model to classsify  dissaster messages categories'''
    pipeline =  Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(
                OneVsRestClassifier(
                    GradientBoostingClassifier(
                        n_estimators = 300,
                        random_state = 0
            ), n_jobs=-1), n_jobs=-1))
    ])
    parameters = {'clf__estimator__estimator__max_depth' : [3],
                 'clf__estimator__estimator__learning_rate': [0.1, 0.2],
                 'tfidf__use_idf': [True],
                 'vect__max_features': [None],
                 'vect__max_df': [1.0]
                 }
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''prints f1 score, precision and recall for each output
     category of the dataset.'''
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, columns=Y_test.columns)
    for col in Y_test.columns:
        print(col)
        print(classification_report(Y_test[col], Y_pred[col]))


def save_model(model, model_filepath):
    dump(model, model_filepath) 


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        #with active_session():
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
