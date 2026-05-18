# tfidf.py
import numpy as np
import math

class TFIDFVectorizer:
    def __init__(self):
        self.vocab = {}  # Maps words to column indices
        self.idf = {}    # Stores the IDF score for each word

    def _tokenize(self, text):
        """Helper to lowercase and split text. In production, you'd strip punctuation here."""
        return text.lower().split()

    def fit(self, corpus):
        """
        Learns the vocabulary dictionary and computes IDF scores.
        :param corpus: A list of string documents.
        """
        total_docs = len(corpus)
        document_frequency = {}

        # 1. Build Vocabulary and Document Frequency (DF)
        for document in corpus:
            words = self._tokenize(document)
            # Use a set because we only care IF the word is in the doc, not how many times
            unique_words = set(words)
            
            for word in unique_words:
                # Increment document frequency
                document_frequency[word] = document_frequency.get(word, 0) + 1
                
                # Add to vocabulary mapping if new
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)

        # 2. Calculate Inverse Document Frequency (IDF) with smoothing
        for word, freq in document_frequency.items():
            # Smoothing prevents division by zero and log(0)
            self.idf[word] = math.log((1 + total_docs) / (1 + freq)) + 1

    def transform(self, corpus):
        """
        Transforms documents to a document-term matrix.
        """
        # Create an empty matrix of shape (documents, vocabulary_size)
        tfidf_matrix = np.zeros((len(corpus), len(self.vocab)))

        for row_idx, document in enumerate(corpus):
            words = self._tokenize(document)
            total_words_in_doc = len(words)
            
            # Count Term Frequency (TF)
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

            # Calculate TF-IDF for this document
            for word, count in word_counts.items():
                if word in self.vocab:
                    col_idx = self.vocab[word]
                    tf = count / total_words_in_doc
                    
                    tfidf_matrix[row_idx, col_idx] = tf * self.idf[word]

        # 3. L2 Normalization (Standard practice so long and short docs are comparable)
        norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
        # Add epsilon to prevent dividing by zero for empty documents
        tfidf_matrix = tfidf_matrix / (norms + 1e-8)

        return tfidf_matrix


# --- Quick Test ---
if __name__ == "__main__":
    # A tiny corpus to prove the math works
    corpus = [
        "the AI engineer built a neural network",
        "the software engineer wrote python code",
        "the AI model learned python"
    ]

    print("🧠 Initializing TF-IDF Vectorizer...")
    vectorizer = TFIDFVectorizer()
    
    print("📚 Fitting corpus to learn vocabulary and IDF weights...")
    vectorizer.fit(corpus)
    
    print(f"\nVocabulary Size: {len(vectorizer.vocab)} unique words")
    
    print("\n🔍 Transforming text into mathematical tensors...")
    tfidf_matrix = vectorizer.transform(corpus)
    
    # Let's check the weights of specific words
    the_idx = vectorizer.vocab["the"]
    engineer_idx = vectorizer.vocab["engineer"]
    neural_idx = vectorizer.vocab["neural"]
    
    print("\n📊 Feature Analysis:")
    print(f"Weight of 'the' in Doc 1: {tfidf_matrix[0, the_idx]:.4f} (Low, because it's everywhere)")
    print(f"Weight of 'engineer' in Doc 1: {tfidf_matrix[0, engineer_idx]:.4f} (Medium, appears in 2 docs)")
    print(f"Weight of 'neural' in Doc 1: {tfidf_matrix[0, neural_idx]:.4f} (High, incredibly rare!)")