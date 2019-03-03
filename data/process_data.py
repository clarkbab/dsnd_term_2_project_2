import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Loads the data from CSV and merges into a dataframe.

    Arguments:
        messages_filepath -- the path to the messages CSV file.
        categories_filepath -- the path to the categories CSV file.
    """
    # Load messages and categories.
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge the 2 dataframes.
    df = messages.merge(categories, on='id')
    
    # Split the categories column into many columns.
    categories = pd.DataFrame(df['categories'].str.split(';', expand=True))
    
    # Set the column names for the categories.
    cols = categories.loc[0].str.split('-', expand=True)[0].values
    categories.columns = cols
    
    # Set each value to a numeric.
    for category in categories:
        # Set each value to the last character of the string.
        categories[category] = categories[category].str[-1]

        # Convert to numeric.
        categories[category] = pd.to_numeric(categories[category])
        
    # Replace the old categories column.
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    return df

def clean_data(df):
    """Removes duplicates from the data.

    Arguments:
        df -- the dataframe.
    """
    # Remove duplicates.
    return df.drop_duplicates()

def save_data(df, database_filename):
    """Saves the cleaned data to an SQLite database.

    Arguments:
        df -- the dataframe to save.
        database_filename -- the name of the database file.
    """
    engine = create_engine(f"sqlite:///{database_filename}")
    conn = engine.connect()
    conn.execute('DROP TABLE messages')
    df.to_sql('messages', engine, index=False)


def main():
    """Runs the data processing script.
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