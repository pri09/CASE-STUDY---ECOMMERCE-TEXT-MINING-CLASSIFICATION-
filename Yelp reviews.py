import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
yelp = pd.read_csv('C:/Users/hp/Desktop/yelp.csv')
yelp.head()
yelp.info()
yelp['Text Length'] = yelp['text'].apply(len)
yelp.corr()
import string
from nltk.corpus import stopwords
def text_preprocess(st):
    no_punc = [i for i in st if i not in string.punctuation]
    no_punc = ''.join(no_punc)
    for i in no_punc:
        new_st = no_punc.rstrip('\n\n')
    return [i for i in new_st.split() if i not in stopwords.words('english')]
yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars']==2) | (yelp['stars']==3) | (yelp['stars']==4) | (yelp['stars']==5)]
yelp_class['stars'].value_counts()
yelp_class.info()
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_preprocess)),
    ('Classifier', MultinomialNB())
])
from sklearn.model_selection import train_test_split
X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
pipeline.fit(X_train,y_train)
predictions = pipeline.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
df = pd.DataFrame(data=X_test)
df['Predictions'] = predictions
df.to_csv('C:/Users/hp/Desktop/Output.csv', index=False)