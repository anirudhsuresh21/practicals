import nltk
from nltk.corpus import stopwords

#Define the documents
document1 = "The quick brown fox jumped over the lazy fox"
document2 = "The lazy dog slept in the sun"

#Get the stopwords for Lnglish language from NLTK
nltk.download('stopwords')
stopWords = stopwords.words('english')

#Step 1: Tokenize the documents
tokens1 = document1.lower().split()
tokens2 = document2.lower().split()

#Combine the tokens into a list of unique terms
terms = list(set(tokens1 + tokens2))

#Step 2: Build the inverted index
inverted_index = {}
occ_num_doc1 = {}
occ_num_doc2 = {}

#For each term, find the documents that contain it
for term in terms:
    if term in stopWords:
        continue
    documents = []
    if term in tokens1:
        documents.append("Document 1")
        occ_num_doc1[term] = tokens1.count(term)
    if term in tokens2:
        documents.append("Document 2")
        occ_num_doc2[term] = tokens2.count(term)
    inverted_index[term] = documents

#Step 3: Print the inverted index
for term, documents in inverted_index.items():
    print(term, "->", end=" ")
    for doc in documents:
        if doc == "Documents 1":
            print(f"{doc} ({occ_num_doc1.get(term,0)}),",end=" ")
        else:
            print(f"{doc} ({occ_num_doc2.get(term,0)}),",end=" ")
    print()

####################################################################
# import os
# import string
# from collections import defaultdict

# def preprocess_text(text):
#     text = text.lower()
#     text = text.translate(str.maketrans("","", string.punctuation))
#     return text.split()

# def build_inverted_index(document):
#     inverted_index = defaultdict(set)
#     for doc_id, text in document.items():
#         words = preprocess_text(text)
#         for word in words:
#             inverted_index[word].add(doc_id)
#     return inverted_index

# def search(inverted_index, query):
#     query_terms = preprocess_text(query)
#     result_set = None
#     for term in query_terms:
#         if term in inverted_index:
#             if result_set is None:
#                 result_set = inverted_index[term]
#             else:
#                 result_set = result_set.intersection(inverted_index[term])
#         else:
#             return result_set()
#     return result_set

# documents = {
#     1:"Information retrieval is an essential aspect of searchj engines.",
#     2:"The field of information retrieval focuses ion algorithm.",
#     3:"Search engines use retrieval techniques to improve perfromance.",
#     4:"Deep learning models are used for information retrieval tasks.",
# }

# inverted_index = build_inverted_index(documents)

# query = "retrieval"
# result = search(inverted_index, query)

# print(f"Documents containing the query '{query}': {sorted(result)}")
