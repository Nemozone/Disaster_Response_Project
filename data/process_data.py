import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
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
    
    # drop all the missing values:
    df.dropna(inplace=True)
   
    # replace categories column in df with new category columns.
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1, sort=False)
    
    return df
    


def clean_data(df):
    df.dropna(inplace=True)     
    #Remove duplicates
    df = df.drop_duplicates()


def save_data(df, database_filepath):
    engine = create_engine(database_filepath)
    df.to_sql('DisasterResponse.db', engine, if_exists='replace', index=False)  


def main():
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