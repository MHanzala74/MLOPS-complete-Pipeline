import os
import logging
import nltk
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import pandas as pd
nltk.download('stopwords')
nltk.download('punkt')

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def transform_text(text):
    """Transform the input text by converting it into lowercase, tokenizing, removing stopswords and punctuation and stemming"""
    ps = PorterStemmer()
    # Convert into lowercase
    text = text.lower()
    # Tokenize the text 
    text = nltk.word_tokenize(text)
    # Remove non alpha numeric tokens 
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    # Stem the words 
    text = [ps.stem(word) for word in text]
    return " ".join(text)

def process_df(df,text_column='text',target_column = 'target'):
    """Preprocess the Dataframe by encoding the target column removing duplicate and transforming the text column"""
    try:
        logger.debug('Starting preprocessing the DataFrame')
        # Encode the target column 
        encode = LabelEncoder()
        df[target_column] = encode.fit_transform(df[target_column])
        logger.debug("Target column decode")

        df = df.drop_duplicates(keep='first')
        logger.debug("Duplicates removed")

        df.loc[:,text_column] = df[text_column].apply(transform_text)
        logger.debug("Text columns transformed")
        return df
    except KeyError as e:
        logger.debug("Column not found %s",e)
        raise
    except Exception as e:
        logger.debug("Error during text normalization %s",e)
        raise

def main(text_column = 'text',target_column= 'target'):
    """Main function to load raw data preprocess it and save the processed data"""
    try:
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded Properly')

        train_processed_data = process_df(train_data,text_column,target_column)
        test_processed_data = process_df(test_data,text_column,target_column)

        data_path = os.path.join("./data","interim")
        os.makedirs(data_path,exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path,'train_processed.csv'),index=False)
        test_processed_data.to_csv(os.path.join(data_path,'test_processed.csv'),index=False)
        logger.debug("Processed data save to %s",data_path)

    except FileNotFoundError as e:
        logger.debug("File not found: %s",e)
    except pd.errors.EmptyDataError as e:
        logger.debug("No data: %s",e)
    except Exception as e:
        logger.error("Failed to complete the data transformation process: %s",e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()





