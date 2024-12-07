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

# Using transformers

![image](https://github.com/user-attachments/assets/0d97f0a1-c555-4d87-a3c9-a0829fe32c95)

O/P:

![image](https://github.com/user-attachments/assets/255ef85f-414b-4ae9-b0ad-4211a3bf882b)

# BOW

Bag of Words (BOW) is a widely used method for text representation in natural language processing (NLP). It is a simple and effective technique to convert text data into numerical vectors that machine learning models can understand.


![image](https://github.com/user-attachments/assets/69433f65-3168-4b34-bab3-26af3c5b4739)

O/P:
![image](https://github.com/user-attachments/assets/7d9eb77b-8d6c-4b2d-98a2-c74f3bbad482)

# Binary BOW
Binary Bag of Words (Binary BOW) is a variant of the Bag of Words (BOW) model where instead of storing word counts, you store binary values (0 or 1) to indicate the presence or absence of words in a document. If the sentence has more than one words. it will represent as 1 only.

vectorizer = CountVectorizer(binary=True)

Advantage-- Simple a& intuitive, Fixed size I/P.
Disadvantage is Spare matrix and ordering of the word completely changed, out of vocub, semantic meaning cannot be captured.

# TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure used to evaluate the importance of a word in a document relative to a collection of documents (corpus). Unlike the Bag of Words (BOW) model, which considers raw word counts, TF-IDF weighs terms based on their frequency in a document and how unique they are across the entire corpus.

![image](https://github.com/user-attachments/assets/56e66224-c3d9-4c44-8af1-97d045166a26)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

Advantage-- Simple a& intuitive, Fixed size I/P, semantic meaning can be captured.
Disadvantage is Spare matrix and ordering of the word completely changed, out of vocub.

# N- Grams

vectorizer = CountVectorizer(binary=True,ngram_range(2,3)) # It will check both bi gram and trigram.






