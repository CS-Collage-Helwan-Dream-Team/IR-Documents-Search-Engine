import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict
import math
import re
import numpy as np
from tabulate import tabulate  # Import tabulate

nltk.download('punkt')

# Read 10 files (.txt)
def read_files(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    data = {}
    for file in files:
        with open(os.path.join(directory, file), 'r') as f:
            data[file] = f.read()
    return data

# Apply tokenization and Step 3: Apply stemming
def tokenize_and_stem(text):
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

# Build positional index
def build_positional_index(data):
    positional_index = defaultdict(lambda: defaultdict(list))

    for doc, text in data.items():
        tokens = tokenize_and_stem(text)
        for position, term in enumerate(tokens):
            positional_index[term][doc].append(position + 1)

    return positional_index

# Compute term frequency
def compute_term_frequency(positional_index):
    term_frequency = defaultdict(lambda: defaultdict(int))

    for term, docs in positional_index.items():
        for doc, positions in docs.items():
            term_frequency[term][doc] = len(positions)

    return term_frequency

# Compute IDF
def compute_idf(data, term, positional_index):
    total_docs = len(data)
    doc_count = sum(1 for doc, positions in positional_index[term].items() if positions)
    return math.log10(total_docs / doc_count) if doc_count != 0 else 0

# Compute TF-IDF matrix
def compute_tf_idf(term_frequency, idf):
    tf_idf_matrix = defaultdict(lambda: defaultdict(float))

    for term, docs in term_frequency.items():
        for doc, tf in docs.items():
            tf_idf_matrix[term][doc] = tf * idf[term]

    return tf_idf_matrix

# Compute similarity between the query and matched documents
def compute_similarity(query_vector, doc_vector):
    # Using cosine similarity
    dot_product = np.dot(query_vector, doc_vector)
    norm_query = np.linalg.norm(query_vector)
    norm_doc = np.linalg.norm(doc_vector)

    return dot_product / (norm_query * norm_doc) if norm_query != 0 and norm_doc != 0 else 0

# Rank documents based on similarity score
def rank_documents(query_vector, tf_idf_matrix):
    scores = {}
    for doc in tf_idf_matrix['terms'].keys():
        doc_vector = np.array([tf_idf_matrix[term][doc] for term in tf_idf_matrix.keys() if term != 'terms'])
        similarity = compute_similarity(query_vector, doc_vector)
        scores[doc] = similarity

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# Allow users to write a phrase query
def phrase_query(positional_index, query):
    terms = tokenize_and_stem(query)
    matched_docs = set(positional_index[terms[0]])

    for term in terms[1:]:
        matched_docs.intersection_update(positional_index[term])

    return matched_docs

# Allow users to write boolean query
def boolean_query(positional_index, query):
    query = query.lower()
    query_tokens = tokenize_and_stem(query)

    stack = []
    operator_stack = []

    for token in query_tokens:
        if token == 'and':
            operator_stack.append('and')
        elif token == 'or':
            operator_stack.append('or')
        elif token == 'not':
            operator_stack.append('not')
        else:
            if token in positional_index:
                stack.append(set(positional_index[token].keys()))
            else:
                stack.append(set())

    while operator_stack:
        operator = operator_stack.pop()
        set2 = stack.pop()

        if not stack:
            return list(set2)

        set1 = stack.pop()

        if operator == 'and':
            result = set1.intersection(set2)
        elif operator == 'or':
            result = set1.union(set2)
        elif operator == 'not':
            result = set1.difference(set2)

        stack.append(result)

    if stack:
        return list(stack[0])
    else:
        return []

# Example usage
data_directory = r"C:\Users\Abdelrahman Mostafa\Desktop\IR\lastIR_project\lastIR_project\files"
data = read_files(data_directory)
positional_index = build_positional_index(data)
term_frequency = compute_term_frequency(positional_index)

# Compute IDF for each term
idf = {term: compute_idf(data, term, positional_index) for term in positional_index}

# Compute TF-IDF matrix
tf_idf_matrix = compute_tf_idf(term_frequency, idf)

# Print positional_index in a table
# print("Positional Index:")
# print(tabulate(positional_index, headers="keys", tablefmt="pretty"))

# Example phrase query
query_phrase = "antony"
matched_docs_phrase = phrase_query(positional_index, query_phrase)
print("Matched Documents (Phrase Query):", matched_docs_phrase)

# Example boolean query
query_boolean = "antony AND brutus AND caeser AND cleopatra AND mercy AND worser"
matched_docs_boolean = boolean_query(positional_index, query_boolean)
print("Matched Documents (Boolean Query):", matched_docs_boolean)

# Example computation of IDF for each term
print("IDF for each term:", idf)

# Example computation of TF-IDF matrix
print("TF-IDF Matrix:")
print(tabulate(tf_idf_matrix, headers="keys", tablefmt="pretty"))

# Example computation of similarity and ranking
query_vector = np.array([tf_idf_matrix[term]['query'] for term in tf_idf_matrix.keys() if term != 'terms'])
ranking = rank_documents(query_vector, tf_idf_matrix)
print("Ranked Documents based on Similarity:")
for doc, score in ranking:
    print(f"Document: {doc}, Similarity Score: {score}")
