from operator import itemgetter
from sklearn.metrics.pairwise import cosine_similarity

from bns import BNS


BNS_VECTORIZER = BNS()

documents = ['please book flights to mumbai', 'need a flight to goa', 'airline price for 2 adult', 'plan a trip to goa', 
             'book a taxi for me', 'book ola for home', 'show uber around me', 'nearby gym around me', 'nearby by temple',
             'i want to know nearby cinema hall in mumbai']

categories = ['book_flight', 'book_flight', 'book_flight', 'book_flight', 'book_taxi', 'book_taxi', 'book_taxi', 'nearby', 
              'nearby', 'nearby']

BNS_VECTORIZER.fit(documents, categories)
# print("vectors: %s" %(BNS_VECTORIZER.vectors))


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