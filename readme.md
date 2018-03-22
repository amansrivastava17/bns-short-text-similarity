## BNS Vectorizer - Improved TF-IDF for shorter text 

Bi-normal Separation is a popular method to score textual data importance against its belonging category, it can efficiently find out important keywords in a document and assign a weighted positive score, also provide negative scoring for unimportant word for a document.

Below are the description of variables used to calculate Bi-normal separation score for a
word for each category (or classes).



### Why BNS Better than TF-IDF?

Due to the short length of the documents, the existing approaches of  TF-IDF and other term frequency bases approaches, did not perform well as  there are usually no words that occur more than once per document, so we need to use an approach that does not rely on term frequency within the document.

BNS overcomes this problem as it  assign weights to each term based on their occurance in positive and negative categories (or classes). A term occurs often in the positive samples and seldom in negative ones, will get a high BNS weight.

Also as *idf* have a general value for term across categories,  *bns* assign different weightage score for term  in different category.  

### Formula to calculate BNS:

- *pos* = number of positive training cases, typically minority,

- *neg* = number of negative training cases,

- *tp* = number of positive training cases containing word,

- *fp* = number of negative training cases containing word,

- *fn* = *pos* - *tp*,

- *tn* = *neg* - *fp*,

- *tpr* (true positive rate) = P(word | positive class) = *tp*/*pos**

- *fpr* (false positive rate)  = P(word | negative class) = *fp*/*neg*,

- *bns* (Bi-Normal Separation) =  F^(-1)(tpr)  –  F^(-1)(fpr)

  *F^(-1) is  the  inverse  Normal  cumulative  distribution  function*

### Usage:

##### Create BNS Vectorizer

```python
from bns import BNS
documents = ['please book flights to mumbai', 'need a flight to goa', 'airline price for 			   2 adult', 'plan a trip to goa', 'book a taxi for me', 'book ola for home',              'show uber around me', 'nearby gym around me', 'nearby by temple',
             'i want to know nearby cinema hall in mumbai']

categories = ['book_flight', 'book_flight', 'book_flight', 'book_flight', 'book_taxi', 				  'book_taxi', 'book_taxi', 'nearby', 'nearby', 'nearby']

BNS_VECTORIZER = BNS()
BNS_VECTORIZER.fit(documents, categories)
```

##### Calculate Cosine similarity

```python
from operator import itemgetter
from sklearn.metrics.pairwise import cosine_similarity

test_documents = ['book me a flight please']
test_bns_vectors = BNS_VECTORIZER.transform(test_documents)

# Lets find most similar sentence and category for given test document
results = []
for category in test_bns_vectors.keys():
    vector = test_bns_vectors[category]
    category_trained_sentence_vectors = BNS_VECTORIZER.vectors[category]
    category_trained_sentence = BNS_VECTORIZER.sentences_category_map[category]
    cosine_scores = cosine_similarity(vector, category_trained_sentence_vectors)[0]
    for score, sent in zip(cosine_scores, category_trained_sentence):
        results.append({'match_sentence':sent, 'category': category, 'score':score})

results = sorted(results, key=itemgetter('score'), reverse=True)
for each in results:
    print each
```

Above similarity method might not produce good results as there are no preprocessing involved, here you can refer to my previous repository to perform various text preprocessing involved before sending documents for bns vectorizer creation.

**link** : [text preprocessing python ](https://github.com/amansrivastava17/text-preprocess-python)

*There are still lots of improvement needed to compute similarity for shorter sentences, you must try the above methods and let me know if you have any improvements and suggestions*

 Thanks !!