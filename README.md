# NLP_Text-Preprocessing

# Tokenization:
Tokenization is the process of splitting a text into smaller units called tokens, which could be words, sentences, or subwords. This is a crucial preprocessing step in Natural Language Processing (NLP) because it breaks down raw text into structured components for further analysis.

nltk.download('punkt')

from nltk.tokenize import word_tokenize,sent_tokenize,TreebankWordTokenizer
word_tokenize(text) , sent_tokenize(text), tokens = list(text) (character tokenization) , text.split()-white space tokenization, 

tokenizer = TreebankWordTokenizer()
tokens = tokenizer.tokenize(text)-- It will tokenize the words in this way. For example we have I'm /It's--> o/p will be ['I', "'m",'It', "'s"]

Subword Tokenization (Using BERT tokenizer)
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer.tokenize(text)

tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(text)

# Stemming:
Try to find the base of the specific word/base stem. SOme times we dont have meaning for the word(biggest disadvantage). It is fast(adv)
finally , final, finalized-> After stemming the word look like fina(no meaning)


from nltk.stem import PorterStemmer,SnowballStemmer

porter_stemmer = PorterStemmer(), snowball_stemmer = SnowballStemmer("english")

stemmed_words = [porter_stemmer.stem(word) for word in words] / stemmed_words = [snowball_stemmer.stem(word) for word in words]

# Lemmitization:
Give meaning ful word(adv), Slow(stem for entire dictionary

from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()---lemmatized_words = [lemmatizer.lemmatize(word) for word in words]/ lemmatizer.lemmatize("going",pos='v')


# Stopwords:
Remove the common words which did not impact

ntlk.download('stopwords').. IN place of english we can use any language. we can setup soem custom stopwords.

stop_words = set(stopwords.words('english'))--filtered_words = [word for word in words if word.lower() not in stop_words]

# Parts of speech Taging:
TO IDENTIFY THE Pos Tag

nltk.pos_tag(words)


# Named Entity Recognition:
 It is a technique used in Natural Language Processing (NLP) to identify and classify named entities (such as names of people, organizations, locations, dates, etc.) in a given text.

 # Download necessary NLTK resources
nltk.download('punkt'),nltk.download('maxent_ne_chunker'),nltk.download('words')

nltk.ne_chunk(pos_tags) # applied on the pos tags

# For Spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_)
    
o/p will look like these

Barack Obama PERSON,Honolulu GPE,Hawaii GPE,August 4, 1961 DATE

Using transformers


![image](https://github.com/user-attachments/assets/0d97f0a1-c555-4d87-a3c9-a0829fe32c95)



