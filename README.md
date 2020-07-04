# Disaster Response Classification

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Running Locally](#files)
4. [File Descriptions](#files)
5. [Results](#results)

## Installation <a name="installation"></a>

This app has only been tested in python3.6.3. To install, create a python 3.6.3 virtual environment, activate it, and install the dependencies specified in the `requirements.txt` file :

```bash
virtualenv --python=<path to python 3.6.3>  <path to virtualenv/>
source <path to virtualenv>/bin/activate
pip install -r requirements.txt
```


## Project Motivation<a name="motivation"></a>

Can we predict whether there is a disaster going on in a place from twitter messages? This project was aimed to create a classifier for the disaster response messages contained in `/data/disaster_messages.csv` and the categories contained in `data/disaster_categories.csv`. A GradientBoostingClassifier was used for this task.

## Running locally <a name="local"></a>

* To clean the data and save it in a database run:

```bash
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_database.db 

```

* To build and train the classifier run:

```bash
python models/train_classifier.py data/disaster_database.db models/model.pkl 

```

* To run the app locally, go to the app folder and run:

```bash
python run.py
```

Then go to http://0.0.0.0:3001/


## File Descriptions <a name="files"></a>

* `requirements.txt`. This file contains the requirements of the project.
* `/data/disaster_messages.csv`. This file contains disaster messages data. 
* `data/disaster_categories.csv`. This file contains data for the categories of the type of event related to each message.
* `data/process_data.py`. Python script for cleaning and merging the datasets.
* `data/disaster_database.db`. Example of database obtained with process_data script.
* `models/tokenize_messages.py`. Python script for message tokenization.
* `models/train_classifier.py`. Python script for constructing and training a classifier for the messages. Running this script can take several up to three hours.
* `models/model.pkl`. An example of a model created with the train_classifier script
* `app/run.py`. Script for running the application.
* `app/templates`. HTML templates for the app.



## Results<a name="results"></a>

The final app can be consulted here: https://classify-disaster-msg.herokuapp.com