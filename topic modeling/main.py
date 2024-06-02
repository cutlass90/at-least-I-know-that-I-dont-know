import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import LdaModel
from gensim import corpora
from gensim.models import LsiModel

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

def main():
    # Creating example documents
    paths = ['astronomy.txt', 'kyiv_cake.txt', 'social.txt', 'story.txt']
    corpus = [open(path).read() for path in paths]

    clean_corpus = [clean(doc).split() for doc in corpus]

    # Creating document-term matrix
    dictionary = corpora.Dictionary(clean_corpus)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in clean_corpus]
    # LSA model
    lsa = LsiModel(doc_term_matrix, num_topics=4, id2word=dictionary)
    print(lsa.print_topics(num_topics=4, num_words=3))

    # LDA model
    lda = LdaModel(doc_term_matrix, num_topics=4, id2word=dictionary)
    # Results
    print(lda.print_topics(num_topics=4, num_words=3))

if __name__ == "__main__":
    main()


