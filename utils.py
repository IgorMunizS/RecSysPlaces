import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

def readDatabase():
    restaurants = pd.read_csv('Datasets/Restaurants/geoplaces2.csv')
    rating = pd.read_csv('Datasets/Restaurants/new_rating.csv')
    hotels = pd.read_csv('Datasets/Hotels/hotels_review.csv')
    museums = pd.read_csv('Datasets/Museums/museum.csv')
    return restaurants, rating, hotels, museums

def preprocessData(rating):
    ratings_pivot = rating.pivot(index='placeID', columns='userID', values='final_rating')
    ratings_pivot.fillna(0, inplace=True)
    ratings_pivot = ratings_pivot.astype(np.int32)
    ratings_matrix = csr_matrix(ratings_pivot.values)

    return ratings_matrix, ratings_pivot

def buildKnnModel(ratings_matrix):
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(ratings_matrix)

    return model_knn

def get_items_interacted(userID, user_df):
    interacted_items = user_df.loc[userID]['placeID']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])