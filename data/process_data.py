import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Takes two user-provided datasets, merges them, returns result as Pandas DataFrame
    
    Args: 
        messages_filepath (string): filepath to messages data stored as csv file,
        categories_filepath (string): filepath to categories data stored as csv file
		
    Returns: 
        df (Pandas DataFrame): merged and wrangled dataset
    """

    # load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, left_on='id', right_on='id')
    
    # create a dataframe of the 36 individual category columns
    categories = categories.categories.str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = [x.split('-')[0] for x in row]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(object).str.split('-').str[1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(float)
    
    # drop the original categories column from `df`
    del df['categories']
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    return df
    pass

def clean_data(df):
    """Takes Pandas DataFrame and drops duplicates
    
    Args: 
        df (Pandas Dataframe): dataframe containing messages and their tagged categories
		
    Returns: 
        df (Pandas DataFrame): cleaned dataframe
    """

    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df
    pass


def save_data(df, database_filename):
    """Takes data frame and saves as sqlite db with user-provided filename
    
    Args: 
        df (Pandas Dataframe): cleaned dataframe containing messages and their tagged categories
        database_filename (string): user-provided name where to store database
		
    Returns: 
        None
    """

    engine = create_engine('sqlite:///' + str(database_filename))
    df.to_sql('etlOutput', engine, index=False)
    pass  

def main():
    """Merges messages and categories data, cleans resulting dataframe, stores into database
    
    Args: 
        None
		
    Returns: 
        None
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