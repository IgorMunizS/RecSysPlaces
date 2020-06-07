from pathlib import Path

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk import word_tokenize
import pickle

class SentimentAnalysisModel:


    def training(self):
        max_words = 20000

        (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_words)

        max_review_length = 100

        X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
        X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

        embedding_vector_length = 32

        self.model = Sequential()
        self.model.add(Embedding(max_words, embedding_vector_length, input_length=max_review_length))
        self.model.add(LSTM(100))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model.fit(X_train, y_train, epochs=3, batch_size=64)

        # Evaluate the trained model on the test data
        model_scores = self.model.evaluate(X_test, y_test, verbose=0)

        # Print out the accuracy of the model on the test set
        print("Sentiment Analysis accuracy on the test dataset: {0:.2f}%".format(model_scores[1] * 100))

    def save_model(self):
        Path("models/").mkdir(parents=True, exist_ok=True)
        with open('models/sentiment_analysis', 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self):
        with open('models/sentiment_analysis', 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, review):
        word2index = imdb.get_word_index()
        test = []
        for word in word_tokenize(review):
            try:
                test.append(word2index[word])
            except:
                test.append(0)

        test = sequence.pad_sequences([test], maxlen=500)
        print(self.model.predict(test))