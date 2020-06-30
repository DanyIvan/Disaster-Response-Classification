import json
import plotly
import pandas as pd
import numpy as np

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine
import sys
sys.path.append('../models')
from tokenize_messages import tokenize

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/disaster_database.db')
df = pd.read_sql_table('disaster_response', engine)

# load model
model = joblib.load("../models/model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    category_names = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_counts = {}
    for category in category_names:
        count = df[category].sum()
        category_counts[category] = count
    category_counts = {k: v for k, v in sorted(category_counts.items(),
                        key=lambda item: item[1])}

    #extrat some example messages 
    examples = np.random.choice(df.message, 100)  
    short_examples = examples[:10]
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    genre_counts_graph = {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }

    category_counts_graph = {
            'data': [
                Bar(
                    x=list(category_counts.keys()),
                    y=list(category_counts.values())
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'automargin': True
                }
            }
        }

    graphs = [genre_counts_graph, category_counts_graph]
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON,
         examples= examples, short_examples=short_examples)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()