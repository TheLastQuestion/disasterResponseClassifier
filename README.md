# Disaster Response Pipeline Project

#### _Not an official position of the US Govt._

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [The Data](#thedata)
4. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Make sure you have libraries etc. installed
    `pip install -r requirements.txt`


3. Run the following command in the app's directory to run your web app.
    `python run.py`

4. Go to http://0.0.0.0:3001/

## Project Motivation<a name="motivation"></a>

I am currently making my wy through an intensive data science certification course (Udacity). This repository is one of the main projects in that course. It involves writing an ETL pipeline, ML pipeline, then using the products to power a data dashboard deployable on Flask.

## The Data <a name="thedata"></a>

The data are real messages from various mediums collected and tagged by FigureEight. They are a great corpus for learning how to categorize messages by the topics they concern, which can then help organizations and services be routed more efficiently in a disaster scenario.

## File Descriptions <a name="files"></a>

- app
    - templates
        - master.html  # main page of web app
        - go.html  # classification result page of web app
    - run.py  # Flask file that runs app

- data
    - disaster_categories.csv  # data to process *(Not included in this repo, due to size constraints)*
    - disaster_messages.csv  # data to process *(Not included in this repo, due to size constraints)*
    - process_data.py
    - InsertDatabaseName.db   # database to save clean data to

- models
    - train_classifier.py
    - classifier.pkl  # saved model *(Not included in this repo, due to size constraints)*

- README.md

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

I've included the MIT license here to be as permissive as makes sense. It is my understanding that Udacity is in accordance with this licencing (they provided starter code and instruction).
