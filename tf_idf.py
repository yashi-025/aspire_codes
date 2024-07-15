# tf-idf verctorisation example
# tf stands for term frequency (i,e no. of time term t appers upon total numbers of 
# terms in that document)
# whereas idf stands for inverse document frequency (i.e log(total no. of documnets 
# upon no. of documents containing the term t))
#Purpose of tf-idf score: Provides a numerical value representing the importance of a term in a 
# document, considering both its frequency in that document and its rarity across 
# the entire corpus (collection of documents).
#vectorisation here reffers to the coversion of obtained score into matrix where row
#represent document and column represent word.

#example 

from sklearn.feature_extraction.text import TfidfVectorizer
documents = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat and the dog"
]

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = vectorizer.fit_transform(documents)

# Display the TF-IDF matrix
print(tfidf_matrix.toarray())

# Display the feature names (terms)
print(vectorizer.get_feature_names_out())
