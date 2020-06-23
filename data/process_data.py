import sys
import pandas as pd 
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    ''' Loads the data of messages_filepath and categories_filepath,
      and merges these datasets
      INPUT - messages_filepath (str): path to messages file
              categories_filepath (str): path to categories file
      OUTPUT - dataframe with merged messages and 
      '''
    #Read the data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # Merge the data
    df = messages.merge(categories, how='inner', on='id')
    return df


def clean_data(df):
    '''cleans the messages and categories dataset
    INPUT - df : messages and categories dataframe
    OUTPUT - df : cleaned dataset
    '''
    #split categories into separate category columns
    categories = df.categories.str.split(expand=True, pat=';')
    row = categories.iloc[0]
    category_colnames = [x[:-2] for x in row ]
    categories.columns = category_colnames
    #convert category columns to values of 0 or 1
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    # convert column from string to numeric
        categories[column] = categories[column].astype('int')
    # drop the original categories column from `df`
    df.drop(columns='categories', inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    #remove duplicates
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    '''saves cleaned dataset to a database file
    INPUT - df: cleaned dataframe
            database_filename (str) : name of the databse file
    '''
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql(name = 'disaster_response',
               con = engine, 
               index=False, 
               if_exists = 'replace')


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