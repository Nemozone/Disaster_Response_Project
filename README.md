# Disaster Response Pipeline Project
A simple web app which visualize and classifies disaster response messages using machine learning algorithms applied on real messages datasets.

### Table of Content
1. [Instruction](#instruction)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)


## Instructions <a name="instruction"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Project Motivation<a name="motivation"></a>

The goal here is building a model based on a data containing thousands of messages, provided by Figure Eight, that were sent during natural disasters.
These messages were sent either via social media or directly to disaster response organizations. I have built an ETL pipeline that processes message and category data from CSV files, and load them into a SQLite database, which the machine learning pipeline will then read from to create and save a multi-output supervised learning model. The result will be demonstrated as a visualization and an interactive classification of messages through a web app.

## File Descriptions <a name="files"></a>

There are 3 folders:
. app containing run.py and html templates
. data containing csv files and process_data.py
. models containing train_classifier.py

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Figure Eight for the data and Udacity for giving the necessary trainings and provided excelent mentorship to do this project.
