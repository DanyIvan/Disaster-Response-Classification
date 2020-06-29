import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'stopwords'])

def tokenize(text):
    '''cleans text and tokeinizes it'''
    #clean urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    #remove punctuation
    text = re.sub('[^a-zA-Z1-9]', ' ', text)
        
    #tokeinize and lemmatize text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    #remove stopwords
    tokens = [x for x in tokens if x not in stopwords.words('english')]
    
    #remove spaces and convert to lower
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens