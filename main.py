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

#Read 10 files (.txt)
def read_files(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    data = {}
    for file in files:
        with open(os.path.join(directory, file), 'r') as f:
            data[file] = f.read()
    return data

#Apply tokenization and Step 3: Apply stemming
def tokenize_and_stem(text):
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

#Build positional index
def build_positional_index(data):
    positional_index = defaultdict(lambda: defaultdict(list))

    for doc, text in data.items():
        tokens = tokenize_and_stem(text)
        for position, term in enumerate(tokens):
            positional_index[term][doc].append(position + 1)

    return positional_index

#Compute term frequency
def compute_term_frequency(positional_index):
    term_frequency = defaultdict(lambda: defaultdict(int))

    for term, docs in positional_index.items():
        for doc, positions in docs.items():
            term_frequency[term][doc] = len(positions)

    return term_frequency

#Compute IDF
def compute_idf(data, term):
    total_docs = len(data)
    doc_count = sum(1 for doc, positions in positional_index[term].items() if positions)
    return math.log10(total_docs / doc_count) if doc_count != 0 else 0

#Compute TF-IDF matrix
def compute_tf_idf(term_frequency, idf):
    tf_idf_matrix = defaultdict(lambda: defaultdict(float))

    for term, docs in term_frequency.items():
        for doc, tf in docs.items():
            tf_idf_matrix[term][doc] = tf * idf[term]

    return tf_idf_matrix

#Allow users to write a phrase query
def phrase_query(positional_index, query):
    terms = tokenize_and_stem(query)
    matched_docs = set(positional_index[terms[0]])

    for term in terms[1:]:
        matched_docs.intersection_update(positional_index[term])

    return matched_docs

#Allow users to write boolean query
def boolean_query(positional_index, term_frequency, query):
    query_terms = re.findall(r'\bAND\b|\bOR\b|\bNOT\b|\(|\)|\b\w+\b', query)
    stack = []
    operators = {'AND', 'OR', 'NOT'}

    try:
        for term in query_terms:
            term_lower = term.lower()
            if term == '(' or term == 'NOT':
                stack.append(term)
            elif term == 'AND' or term == 'OR':
                stack.append(term)
            elif term == ')':
                while stack and stack[-1] not in {'(', 'AND', 'OR'}:
                    operator = stack.pop()
                    if operator == 'AND':
                        stack.append(set(stack.pop()) & set(stack.pop()))
                    elif operator == 'OR':
                        stack.append(set(stack.pop()) | set(stack.pop()))
                    elif operator == 'NOT':
                        stack.append(set(list(matched_docs_boolean) - set(stack.pop())))
                if stack and stack[-1] in {'(', 'AND', 'OR'}:
                    stack.pop()  # Remove the '(' or boolean operator
            else:
                stemmed_term = tokenize_and_stem(term_lower)[0]  # Apply stemming
                print("Term found in query:", stemmed_term)
                stack.append(set(positional_index[stemmed_term]))

        while stack:
            operator = stack.pop()
            if operator == 'AND':
                stack.append(set(stack.pop()) & set(stack.pop()))
            elif operator == 'OR':
                stack.append(set(stack.pop()) | set(stack.pop()))
            elif operator == 'NOT':
                stack.append(set(list(matched_docs_boolean) - set(stack.pop())))

        return stack[0] if stack else set()

    except (IndexError, KeyError):
        print("Error: Malformed boolean query or term not found. Please check your query syntax and terms.")
        return set()


#Compute similarity between the query and matched documents
def compute_similarity(query_vector, doc_vector):
    dot_product = np.dot(query_vector, doc_vector)
    query_norm = np.linalg.norm(query_vector)
    doc_norm = np.linalg.norm(doc_vector)

    if query_norm == 0 or doc_norm == 0:
        return 0.0

    similarity = dot_product / (query_norm * doc_norm)
    return similarity

#Rank documents based on similarity score
def rank_documents(query_vector, matched_docs, tf_idf_matrix):
    ranked_documents = []

    for doc in matched_docs:
        doc_vector = np.array([tf_idf_matrix[term][doc] for term in tf_idf_matrix.keys()])
        similarity_score = compute_similarity(query_vector, doc_vector)
        ranked_documents.append((doc, similarity_score))

    ranked_documents.sort(key=lambda x: x[1], reverse=True)
    return ranked_documents

# Example usage
data_directory = r"C:\Users\Abdelrahman Mostafa\Desktop\IR\lastIR_project\lastIR_project\files"
data = read_files(data_directory)
positional_index = build_positional_index(data)
term_frequency = compute_term_frequency(positional_index)

# #print positional_index in a table
# print("Positional Index:")
# print(tabulate(positional_index, headers="keys", tablefmt="pretty"))


# # Example phrase query
# query_phrase = "your phrase query"
# matched_docs_phrase = phrase_query(positional_index, query_phrase)
# print("Matched Documents (Phrase Query):", matched_docs_phrase)

# Example boolean query
query_boolean = "antony OR brutus OR calpurnia"
matched_docs_boolean = boolean_query(positional_index, term_frequency, query_boolean)
print("Matched Documents (Boolean Query):", matched_docs_boolean)

# # Example computation of IDF for each term
# idf = {term: compute_idf(data, term) for term in positional_index}
# print("IDF for each term:", idf)

# # Example computation of TF-IDF matrix
# tf_idf_matrix = compute_tf_idf(term_frequency, idf)
# print("TF-IDF Matrix:")
# print(tabulate(tf_idf_matrix, headers="keys", tablefmt="pretty"))

# # Example usage with similarity and ranking
# # Assuming you have a query and matched documents from the boolean query example
# query_vector = np.array([tf_idf_matrix[term][list(data.keys())[0]] for term in tf_idf_matrix.keys()])
# ranked_documents = rank_documents(query_vector, matched_docs_boolean, tf_idf_matrix)

# # Display the ranked documents in a table
# table_data = []
# for rank, (doc, similarity_score) in enumerate(ranked_documents, start=1):
#     table_data.append([rank, doc, similarity_score])

# # Define headers for the table
# headers = ["Rank", "Document ID", "Similarity Score"]

# # Print the table
# print("Ranked Documents:")
# print(tabulate(table_data, headers=headers, tablefmt="pretty"))
