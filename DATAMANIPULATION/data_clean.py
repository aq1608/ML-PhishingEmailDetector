import re
from pandas import read_csv, DataFrame
from functools import reduce
from nltk import word_tokenize, download, pos_tag

from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer

download('punkt_tab', quiet=True)
download('stopwords', quiet=True)
download('averaged_perceptron_tagger', quiet=True)
download('wordnet', quiet=True)
download('omw-1.4', quiet=True)
count = 0

# <---------------------------- Writing into files function ---------------------------->
def write_data(words: DataFrame):
    """
    Fills all NA values with empty string, 
    Writes DataFrame into the cleandata files
    """
    try:
        words.fillna('', inplace = True)
    except KeyError:
        print('KeyError')
    except Exception as e:
        print(e)
    finally:
        if count == 0:
            words.to_csv(r'CLEANDATA\data-1.csv',index=False,mode='a')
        elif count > 1:
            words.to_csv(r'CLEANDATA\data-2.csv',index=False,mode='a',header=False)
        else:
            words.to_csv(r'CLEANDATA\data-2.csv',index=False,mode='a')


# <---------------------------- Replacing text urls function --------------------------->
def find_url(text: str):
    """
    checks if url exists in text
    """
    urls = re.findall(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})', text)
    return urls 

def replace_url(text: str):
    """
    replaces url to empty string
    """
    return re.sub(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})', '', text)

def replace_whitespaces(text: str):
    """
    removes blank spaces
    """
    return re.sub(r'[\r\n]+', ' ', text)

def clean_text(body: DataFrame):
    """
    Finds urls in email body, 
    Replaces urls with empty string, 
    Calls functions to normalise email body and email subject
    Returns cleaned dataset
    """
    url_dict = {}
    body.fillna('',inplace=True)

    try:
        for i in range(body.shape[0]):

            urls = find_url(body.loc[i,"body"])

            if urls != []: 

                urls = ','.join(urls)                
                url_dict.update({i:urls})
                body.loc[i,"body"] = reduce(lambda x, func: func(x), [replace_url,replace_whitespaces],body.loc[i,"body"])
            
            else: url_dict.update({i:''})

            body.loc[i,"body"] = process_text(body.loc[i,"body"])
            body.loc[i,"subject"] = process_text(body.loc[i,"subject"])
        
        if url_dict != {}:
            body.insert(4, 'url', body.index.map(url_dict))
        
        return body
    
    except Exception as e:
        print(e)


# <-------------------------- Normalising text data functions -------------------------->
def tokenize(input_text: str):
    """
    Convert a string of text to a list with words.
    """
    lowercase = input_text.lower()
    token_list = word_tokenize(lowercase)
    clean_list = [word for word in token_list if len(re.findall(r'^[a-zA-Z0-9]+-?[\w-]*$', word)) == 1]
    
    return clean_list

def remove_stopwords(tokenized_text: list):
    """
    Remove stopwords from a list of words.
    """
    stop_words = stopwords.words('english')
    clean_list = [word for word in tokenized_text if word not in stop_words]
    
    return clean_list

def get_wordnet_pos(treebank_tag):
    """
    Converts a Treebank part-of-speech tag to WordNet.
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize(token_list: list):
    """
    Lemmatizes a list of tokens using WordNet lemmatizer.
    """
    lemmatizer = WordNetLemmatizer()
    tagged_list = pos_tag(token_list)
    
    lemmatized_list = [lemmatizer.lemmatize(word)
                       if get_wordnet_pos(tag) is None
                       else lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag))
                       for word, tag in tagged_list]
    
    return lemmatized_list

def join_list(lst: list):
    """
    Joins the list together by spaces to form a string
    """
    return ' '.join(lst)

def process_text(input_text: str):
    """
    Conbines text normalisation functions into one higher order function
    """
    return reduce(lambda x, func: func(x), [tokenize, remove_stopwords, lemmatize, join_list], input_text)


# <------------------------------ main cleaning function ------------------------------>
def dataset_cleaning(dataset: list):
    """
    Reads the list of datasets provided,
    Remove any duplicates (?)
    Drops unused columns
    Calls functions to clean data
    """
    global count

    for data in dataset:
        df = read_csv(data)
        
        df.drop_duplicates()
        for word in ['date','sender']:
            if word not in df.columns: 
                df.insert(0,word,df.index.map({x:'' for x in range(df.shape[0])}))

        try:
            # for csvs with "receiver" and "date" columns
            df.drop(columns=['receiver','urls'],inplace=True)
                        
        except Exception as e:
            pass

        try:

            new_df = clean_text(df.copy())
            write_data(new_df.copy())

            count += 1
        
        except Exception as e:
            print(e)

