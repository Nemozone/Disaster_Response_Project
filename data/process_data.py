import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads the data from csv files, transforms and 
    concatenates into a single dataframe
    
    Parameters:
        messages_filepath (str): filepath for the 'messages' csv file
        categories_filepath (str): filepath for the 'categories' csv file
    
    Returns:
        df (pandas DataFrame): Returning DataFrame 
    
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='outer', on=['id'])
    # change category dataset format to a dummy-var like shape:
    categories = categories['categories'].str.split(';', expand=True)
    row = categories.loc[0,:]
    category_colnames = row.apply(lambda x: str(x)[:-2])
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].str.split('-').str[1].astype(str)
        categories[column] = pd.to_numeric(categories[column])
    
    
    # replace any non binary values:
    categories.replace(2,1,inplace=True)
    
       
    # replace categories column in df with new category columns.
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1, sort=False)
    
    return df
    


def clean_data(df):
    """
    Returns the cleaned data frame with no missing or duplicated values
    
    Paramenters:
        df (pandas DataFrame): The returned data frame from load_data()
    
    Returns:
        df (pandas DataFrame): cleaned data
    
    """
    df.drop('original',axis=1, inplace=True)
    df.dropna(inplace=True)
    category_colnames = ['related','request','offer','aid_related','medical_help',
                         'medical_products', 'search_and_rescue','security', 'military','child_alone',
                         'water','food','shelter','clothing','money',
                         'missing_people','refugees','death','other_aid','infrastructure_related',
                         'transport', 'buildings', 'electricity', 'tools', 'hospitals',
                         'shops', 'aid_centers', 'other_infrastructure', 'weather_related', 'floods',
                         'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report'
                         ]
    df[category_colnames] = df[category_colnames].astype(int)    
    #Remove duplicates
    df = df.drop_duplicates()
    return df


def save_data(df, database_filepath):
    """
    Loads data into a salite database
    
    Paramenters:
        df (pandas DataFrame): The returned data frame from load_data()
        database_filepath (str): the file path for the database file to be restored
    
    Returns:
        Nothing
    
    """
    engine = create_engine('sqlite:///'+ str(database_filepath))
    df.to_sql('DisasterResponse.db', engine, if_exists='replace', index=False)  


def main():
    """
    Executes all the processing by running all the functions and saves the 
    cleaned data into the adressed database
    
    Paramenters:
        No arguments
    
    Returns:
        Nothing
    
    """

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()