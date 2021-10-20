
from lxml import etree
from nltk import word_tokenize, WordNetLemmatizer, pos_tag
from nltk.corpus import stopwords
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer


xml_path = "news.xml"
tree = etree.parse(xml_path)
root = tree.getroot()
corpus = root[0]
header = []
content = []
stop_words_punct = stopwords.words('english') + list(punctuation) + ['ha', 'wa', 'le']
lemmatizer = WordNetLemmatizer()
for news in corpus:
    n = news[0].text
    c = news[1].text
    header.append(n)
    tokens = word_tokenize(c.lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stop_words_punct]
    tokens = [pos_tag([t])[0][0] for t in tokens if pos_tag([t])[0][1] == "NN"]
    content.append(' '.join(tokens))

vectorizer = TfidfVectorizer(analyzer='word', stop_words=stop_words_punct)
tfidf = vectorizer.fit(content)
for i in range(len(content)):
    tfidf_news = vectorizer.transform([content[i]])
    fdist = list(zip((tfidf_news.toarray().tolist()[0]), tfidf.get_feature_names()))
    fdist.sort(key=lambda x: (x[0], x[1]), reverse=True)
    print(header[i] + ':')
    print(' '.join(t[1] for t in fdist[:5]) + '\n')
