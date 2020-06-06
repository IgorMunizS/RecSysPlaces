import pickle
from pathlib import Path
from utils import buildKnnModel
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import scipy
import sklearn
from rake_nltk import Rake


class PopularityRecommender:

    def __init__(self, items_df=None):
        self.popularity_df = ''
        self.items_df = items_df
        self.model_name = 'Popularity'

    def get_model_name(self):
        return self.model_name

    def train_model(self, user_df):
        self.popularity_df = user_df.groupby('placeID')['final_rating'].sum().sort_values(ascending=False).reset_index()

    def save_model(self):
        Path("models/").mkdir(parents=True, exist_ok=True)
        with open('models/popularity', 'wb') as f:
            pickle.dump(self.popularity_df, f)

    def load_model(self):
        with open('models/popularity', 'rb') as f:
            self.popularity_df = pickle.load(f)

    def recommend_items(self, _, items_to_ignore=[], topn=10, verbose=False):
        # Recommend popular items. Best approach to new users
        recommendations_df = self.popularity_df[~self.popularity_df['placeID'].isin(items_to_ignore)] \
            .sort_values('final_rating', ascending = False) \
            .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left',
                                                          left_on = 'placeID',
                                                          right_on = 'placeID')[['final_rating', 'placeID']]

        return recommendations_df


class CFR:
    '''Collaborative Filtering with Matrix Factorization and SVD'''
    '''User Based'''

    def __init__(self, items_df=None):
        self.items_df = items_df
        self.model_name = 'Collaborative Filtering'
        self.cf_predictions_df = ''





    def get_model_name(self):
        return self.model_name

    def train_model(self, user_train_df):
        # Creating a sparse pivot table with users in rows and items in columns
        users_items_pivot_matrix_df = user_train_df.pivot(index='userID',
                                                               columns='placeID',
                                                               values='final_rating').fillna(0)

        users_items_pivot_matrix = users_items_pivot_matrix_df.values
        users_ids = list(users_items_pivot_matrix_df.index)
        users_items_pivot_sparse_matrix = csr_matrix(users_items_pivot_matrix)
        # The number of factors to factor the user-item matrix.
        NUMBER_OF_FACTORS_MF = 12
        # Performs matrix factorization of the original user item matrix
        U, sigma, Vt = svds(users_items_pivot_sparse_matrix, k=NUMBER_OF_FACTORS_MF)

        sigma = np.diag(sigma)
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)

        all_user_predicted_ratings_norm = (all_user_predicted_ratings - all_user_predicted_ratings.min()) / (
                    all_user_predicted_ratings.max() - all_user_predicted_ratings.min())

        self.cf_predictions_df = pd.DataFrame(all_user_predicted_ratings_norm, columns=users_items_pivot_matrix_df.columns,
                                   index=users_ids).transpose()

    def save_model(self):
        Path("models/").mkdir(parents=True, exist_ok=True)
        with open('models/crf', 'wb') as f:
            pickle.dump(self.cf_predictions_df, f)

    def load_model(self):
        with open('models/crf', 'rb') as f:
            self.cf_predictions_df = pickle.load(f)


    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False) \
            .reset_index().rename(columns={user_id: 'final_rating'})

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['placeID'].isin(items_to_ignore)] \
            .sort_values('final_rating', ascending=False) \
            .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                          left_on='placeID',
                                                          right_on='placeID')[['final_rating', 'placeID']]

        return recommendations_df



class ContentBasedNLP:


    def __init__(self, items_df=None):
        self.item_ids = ''
        self.items_df = items_df
        self.model_name = 'Content-Based'

    def get_item_profile(self, item_id):
        idx = self.item_ids.index(item_id)
        item_profile = self.tfidf_matrix[idx:idx + 1]
        return item_profile

    def get_item_profiles(self, ids):
        item_profiles_list = [self.get_item_profile(x) for x in ids]
        item_profiles = scipy.sparse.vstack(item_profiles_list)
        return item_profiles

    def build_users_profile(self, person_id, interactions_indexed_df):
        interactions_person_df = interactions_indexed_df.loc[person_id]
        user_item_profiles = self.get_item_profiles(interactions_person_df['placeID'])

        user_item_strengths = np.array(interactions_person_df['final_rating']).reshape(-1, 1)
        # Weighted average of item profiles by the interactions strength
        user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0) / np.sum(
            user_item_strengths)
        user_profile_norm = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg)
        return user_profile_norm

    def build_users_profiles(self, user_train_df):
        interactions_indexed_df = user_train_df[user_train_df['placeID'] \
            .isin(self.items_df['placeID'])].set_index('userID')
        user_profiles = {}
        for person_id in interactions_indexed_df.index.unique():
            user_profiles[person_id] = self.build_users_profile(person_id, interactions_indexed_df)
        return user_profiles


    def train_model(self, user_train_df):
        review_df = user_train_df.groupby("placeID")['Review'].apply(lambda x: ' '.join(x)).reset_index()

        stopwords_list = stopwords.words('english')
        # Trains a model whose vectors size is 5000, composed by the main unigrams and bigrams found in the corpus, ignoring stopwords
        vectorizer = TfidfVectorizer(analyzer='word',
                                     ngram_range=(1, 2),
                                     min_df=0.003,
                                     max_df=0.5,
                                     max_features=5000,
                                     stop_words=stopwords_list)

        self.item_ids = review_df['placeID'].tolist()
        self.tfidf_matrix = vectorizer.fit_transform(review_df['Review'])
        self.tfidf_feature_names = vectorizer.get_feature_names()

        self.user_profiles = self.build_users_profiles(user_train_df)

    def save_model(self):
        Path("models/").mkdir(parents=True, exist_ok=True)
        with open('models/content-base-tfidf', 'wb') as f:
            pickle.dump(self.user_profiles, f)

        with open('models/content-base-tfidf-matrix', 'wb') as f:
            pickle.dump(self.tfidf_matrix, f)

        with open('models/content-base-tfidf-ids', 'wb') as f:
            pickle.dump(self.item_ids, f)

    def load_model(self):
        with open('models/content-base-tfidf', 'rb') as f:
            self.user_profiles = pickle.load(f)

        with open('models/content-base-tfidf-matrix', 'rb') as f:
            self.tfidf_matrix = pickle.load(f)

        with open('models/content-base-tfidf-ids', 'rb') as f:
            self.item_ids = pickle.load(f)

    def get_model_name(self):
        return self.model_name

    def _get_similar_items_to_user_profile(self, person_id, topn=1000):
        # Computes the cosine similarity between the user profile and all item profiles
        cosine_similarities = cosine_similarity(self.user_profiles[person_id], self.tfidf_matrix)
        # Gets the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        # Sort the similar items by similarity
        similar_items = sorted([(self.item_ids[i], cosine_similarities[0, i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        similar_items = self._get_similar_items_to_user_profile(user_id)
        # Ignores items the user has already interacted
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))

        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['placeID', 'final_rating']) \
            .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                          left_on='placeID',
                                                          right_on='placeID')[
                ['final_rating', 'placeID']]

        return recommendations_df


class HybridRecommender:
    MODEL_NAME = 'Hybrid'

    def __init__(self, cb_rec_model, cf_rec_model, items_df, cb_ensemble_weight=1.0, cf_ensemble_weight=1.0):
        self.cb_rec_model = cb_rec_model
        self.cf_rec_model = cf_rec_model
        self.cb_ensemble_weight = cb_ensemble_weight
        self.cf_ensemble_weight = cf_ensemble_weight
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Getting the top-1000 Content-based filtering recommendations
        cb_recs_df = self.cb_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, verbose=verbose,
                                                       topn=1000).rename(columns={'final_rating': 'finalRatingCB'})

        # Getting the top-1000 Collaborative filtering recommendations
        cf_recs_df = self.cf_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, verbose=verbose,
                                                       topn=1000).rename(columns={'final_rating': 'finalRatingCF'})

        # Combining the results by contentId
        recs_df = cb_recs_df.merge(cf_recs_df,
                                   how='outer',
                                   left_on='placeID',
                                   right_on='placeID').fillna(0.0)

        # Computing a hybrid recommendation score based on CF and CB scores
        # recs_df['recStrengthHybrid'] = recs_df['recStrengthCB'] * recs_df['recStrengthCF']
        recs_df['finalRatingHybrid'] = (recs_df['finalRatingCB'] * self.cb_ensemble_weight) \
                                       + (recs_df['finalRatingCF'] * self.cf_ensemble_weight)

        # Sorting recommendations by hybrid score
        recommendations_df = recs_df.sort_values('finalRatingHybrid', ascending=False).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                          left_on='placeID',
                                                          right_on='placeID')[
                ['finalRatingHybrid', 'placeID']]

        return recommendations_df



class ContentBasedNLPHotels:

    def __init__(self, place_df):
        self.place_df = place_df


    def train_model(self):

        self.place_df['Key_words'] = ''
        r = Rake()
        key_words = []
        for index, row in self.place_df.iterrows():
            r.extract_keywords_from_text(row['Positive_Review'] + row['Negative_Review'])
            key_words_dict_scores = r.get_word_degrees()
            key_words.append(list(key_words_dict_scores.keys()))

        self.place_df['Key_words'] = key_words

        words_list = []
        for index, row in self.place_df.iterrows():
            words = ' '.join(row['Key_words']) + ' '
            words_list.append(words)
        self.place_df['Bag_of_words'] = words_list

        bg_hotels = self.place_df[['Hotel_Name', 'Bag_of_words']]

        count = TfidfVectorizer()
        count_matrix = count.fit_transform(bg_hotels['Bag_of_words'])
        self.cosine_sim = cosine_similarity(count_matrix, count_matrix)

    def save_model(self):
        Path("models/").mkdir(parents=True, exist_ok=True)
        with open('models/content-base-tfidf-hotels', 'wb') as f:
            pickle.dump(self.cosine_sim, f)


    def load_model(self):
        with open('models/content-base-tfidf-hotels', 'rb') as f:
            self.cosine_sim = pickle.load(f)


    def recommend_items(self, place_id):

        recommended_movies = []
        #     idx = indices[indices == title].index[0]
        score_series = pd.Series(self.cosine_sim[place_id]).sort_values(ascending=False)
        top_10_indices = list(score_series.iloc[1:11].index)


        return top_10_indices


class ContentBasedNLPMuseums:

    def __init__(self, place_df):
        self.place_df = place_df


    def train_model(self):
        self.place_df['Description'].fillna('No description', inplace=True)
        self.place_df['Key_words'] = ''
        r = Rake()
        key_words = []
        for index, row in self.place_df.iterrows():
            r.extract_keywords_from_text(row['Description'])
            key_words_dict_scores = r.get_word_degrees()
            key_words.append(list(key_words_dict_scores.keys()))

        self.place_df['Key_words'] = key_words

        words_list = []
        for index, row in self.place_df.iterrows():
            words = ' '.join(row['Key_words']) + ' '
            words_list.append(words)
        self.place_df['Bag_of_words'] = words_list

        bg_museums = self.place_df[['MuseumName', 'Bag_of_words']]

        count = TfidfVectorizer()
        count_matrix = count.fit_transform(bg_museums['Bag_of_words'])
        self.cosine_sim = cosine_similarity(count_matrix, count_matrix)

    def save_model(self):
        Path("models/").mkdir(parents=True, exist_ok=True)
        with open('models/content-base-tfidf-museums', 'wb') as f:
            pickle.dump(self.cosine_sim, f)


    def load_model(self):
        with open('models/content-base-tfidf-museums', 'rb') as f:
            self.cosine_sim = pickle.load(f)


    def recommend_items(self, place_id):

        recommended_movies = []
        #     idx = indices[indices == title].index[0]
        score_series = pd.Series(self.cosine_sim[place_id]).sort_values(ascending=False)
        top_10_indices = list(score_series.iloc[1:11].index)


        return top_10_indices




class CFRKnn:
    '''Collaborative Filtering with KNN'''
    '''User Based'''

    def __init__(self, items_df=None):
        self.items_df = items_df
        self.model_name = 'Collaborative Filtering Knn'
        self.cf_predictions_df = ''





    def get_model_name(self):
        return self.model_name

    def train_model(self, user_train_df):
        # Creating a sparse pivot table with users in rows and items in columns
        places_user_pivot_matrix_df = user_train_df.pivot(index='placeID',
                                                               columns='userID',
                                                               values='final_rating').fillna(0)

        places_user_pivot_matrix = places_user_pivot_matrix_df.values
        places_ids = list(places_user_pivot_matrix_df.index)
        places_users_pivot_sparse_matrix = csr_matrix(places_user_pivot_matrix)
        # The number of factors to factor the user-item matrix.
        model_knn = buildKnnModel(places_users_pivot_sparse_matrix)

    def save_model(self):
        Path("models/").mkdir(parents=True, exist_ok=True)
        with open('models/crf', 'wb') as f:
            pickle.dump(self.cf_predictions_df, f)

    def load_model(self):
        with open('models/crf', 'rb') as f:
            self.cf_predictions_df = pickle.load(f)


    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False) \
            .reset_index().rename(columns={user_id: 'final_rating'})

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['placeID'].isin(items_to_ignore)] \
            .sort_values('final_rating', ascending=False) \
            .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                          left_on='placeID',
                                                          right_on='placeID')[['final_rating', 'placeID']]

        return recommendations_df