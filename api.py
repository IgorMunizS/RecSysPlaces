#!/usr/bin/env python
# coding=utf-8

import threading
import json
import numpy as np
import pandas as pd
from flask import Flask, session, escape, request, jsonify
from time import sleep
from models import PopularityRecommender, CFR, ContentBasedNLP, HybridRecommender, ContentBasedNLPHotels
from models import ContentBasedNLPMuseums


app = Flask(__name__)
app.config['SECRET_KEY'] = "A0Zr98j/3yX R~XHH!jmN]LWX/,?RT"

placeID = ''
response = ""

popularity_model = ''

@app.route('/restaurants/places', methods=['GET', 'POST'])
def get_all_rest():
    restaurants = pd.read_csv("Datasets/Restaurants/geoplaces2.csv")
    all_restaurants =[]
    for i in range(len(restaurants)):
        r_id = restaurants.iloc[i, 0]
        r_name = restaurants.iloc[i, 4]

        metadata = {"ID": int(r_id), "Restaurant Name": r_name}
        all_restaurants.append(metadata)
    response = {"All Restaurants" : all_restaurants}

    return jsonify(response)

@app.route('/hotels/places', methods=['GET', 'POST'])
def get_all_hotels():
    hotels = pd.read_csv("Datasets/Hotels/hotels_review.csv")
    all_hotels =[]
    for i in range(len(hotels)):
        r_id = hotels.index[i]
        r_name = hotels.iloc[i, 0]

        metadata = {"ID": int(r_id), "Hotel Name": r_name}
        all_hotels.append(metadata)
    response = {"All Hotels" : all_hotels}

    return jsonify(response)

@app.route('/museums/places', methods=['GET', 'POST'])
def get_all_museums():
    museums = pd.read_csv("Datasets/Museums/museum.csv")
    all_museums =[]
    for i in range(len(museums)):
        r_id = museums.index[i]
        r_name = museums.iloc[i, 1]

        metadata = {"ID": int(r_id), "Museum Name": r_name}
        all_museums.append(metadata)
    response = {"All Museums" : all_museums}

    return jsonify(response)

@app.route('/restaurants/best', methods=['GET', 'POST'])
def get_best_restaurants():
    popularity_model = PopularityRecommender()
    popularity_model.load_model()
    mean_rating = popularity_model.recommend_items(0)
    place_df = pd.read_csv('Datasets/Restaurants/geoplaces2.csv')
    tr = []
    for i in range(10):
        top_rated_rest = mean_rating.iloc[i,0]
        t_r_name = place_df[place_df["placeID"] == int(top_rated_rest)]["name"].tolist()[0]
        metadata = {"ID": int(top_rated_rest), "Restaurant Name": t_r_name}
        tr.append(metadata)


    tr_response = {"Top Rated" : tr}

    return jsonify(tr_response)


@app.route('/hotels/best', methods=['GET', 'POST'])
def get_best_hotels():
    place_df = pd.read_csv('Datasets/Hotels/hotels_review.csv')
    best_hotels = place_df[place_df.Reviewer_Score > place_df['Reviewer_Score'].quantile(.25)]
    best_hotels = best_hotels.sort_values('Average_Score', ascending=False)[:10]
    tr = []
    for i in range(10):
        top_rated_rest = best_hotels.index[i]
        t_r_name = best_hotels.iloc[i, 0]
        metadata = {"ID": int(top_rated_rest), "Hotel Name": t_r_name}
        tr.append(metadata)


    tr_response = {"Top Rated" : tr}

    return jsonify(tr_response)

@app.route('/museums/best', methods=['GET', 'POST'])
def get_best_museums():
    place_df = pd.read_csv("Datasets/Museums/museum.csv")
    best_museums = place_df[place_df.ReviewCount > place_df['ReviewCount'].quantile(.25)]
    best_museums = best_museums.sort_values('Rating', ascending=False)[:10]
    tr = []
    for i in range(10):
        top_rated_rest = best_museums.index[i]
        t_r_name = best_museums.iloc[i, 1]
        metadata = {"ID": int(top_rated_rest), "Museum Name": t_r_name}
        tr.append(metadata)


    tr_response = {"Top Rated" : tr}

    return jsonify(tr_response)

@app.route('/restaurants/recommend', methods=['GET', 'POST'])
def get_recommended_places():

    if request.method == 'POST' or request.method == 'GET':
        if request.method == 'POST':
            if request.json:
                data = request.get_json()
                session['userid'] = data['userid']

            else:
                session['userid'] = request.form['userid']

        else:
            session['userid'] = request.args.get('userid')


        u_id = str(escape(session['userid']))
        place_df = pd.read_csv('Datasets/Restaurants/geoplaces2.csv')

        #### Collaborative Filtering Recommender ####
        crf_recommend = []
        cf_model = CFR(place_df)
        cf_model.load_model()
        recommend_df = cf_model.recommend_items(u_id)

        for i in range(len(recommend_df)):
            p_id = recommend_df.iloc[i,0]
            p_name = place_df[place_df["placeID"] == p_id]['name'].tolist()[0]
            metadata = {"placeID": int(p_id), "Name": p_name}
            crf_recommend.append(metadata)

        #### Content Based NLP Recommender ####
        cb_recommend = []
        cb_model = ContentBasedNLP(place_df)
        cb_model.load_model()
        recommend_df = cb_model.recommend_items(u_id)

        for i in range(len(recommend_df)):
            p_id = recommend_df.iloc[i, 0]
            p_name = place_df[place_df["placeID"] == p_id]['name'].tolist()[0]
            metadata = {"placeID": int(p_id), "Name": p_name}
            cb_recommend.append(metadata)

        #### Hybrid Recommender ####
        hb_recommend = []
        hb_model = HybridRecommender(cb_model, cf_model, place_df, cb_ensemble_weight=10, cf_ensemble_weight=100)
        recommend_df = hb_model.recommend_items(u_id)

        for i in range(len(recommend_df)):
            p_id = recommend_df.iloc[i, 0]
            p_name = place_df[place_df["placeID"] == p_id]['name'].tolist()[0]
            metadata = {"placeID": int(p_id), "Name": p_name}
            hb_recommend.append(metadata)

        response_ = {"Collaborative Filtering": crf_recommend, "Content Based NLP": cb_recommend, "Hybrid Recommender": hb_recommend}
        return jsonify(response_)



@app.route('/hotels/recommend', methods=['GET', 'POST'])
def get_recommended_hotels():

    if request.method == 'POST' or request.method == 'GET':
        if request.method == 'POST':
            if request.json:
                data = request.get_json()
                session['id'] = data['id']

            else:
                session['id'] = request.form['id']

        else:
            session['id'] = request.args.get('id')


        id = int(escape(session['id']))
        place_df = pd.read_csv('Datasets/Hotels/hotels_review.csv')

        #### Content Based NLP Recommender ####
        cb_recommend = []
        cb_model = ContentBasedNLPHotels(place_df)
        cb_model.load_model()
        recommend_ids = cb_model.recommend_items(id)

        for i in range(len(recommend_ids)):
            p_id = recommend_ids[i]
            p_name = place_df.iloc[p_id,0]
            metadata = {"placeID": int(p_id), "Name": p_name}
            cb_recommend.append(metadata)



        response_ = {"Content Based NLP": cb_recommend}
        return jsonify(response_)


@app.route('/museums/recommend', methods=['GET', 'POST'])
def get_recommended_museums():

    if request.method == 'POST' or request.method == 'GET':
        if request.method == 'POST':
            if request.json:
                data = request.get_json()
                session['id'] = data['id']

            else:
                session['id'] = request.form['id']

        else:
            session['id'] = request.args.get('id')


        id = int(escape(session['id']))
        place_df = pd.read_csv('Datasets/Museums/museum.csv')

        #### Content Based NLP Recommender ####
        cb_recommend = []
        cb_model = ContentBasedNLPMuseums(place_df)
        cb_model.load_model()
        recommend_ids = cb_model.recommend_items(id)

        for i in range(len(recommend_ids)):
            p_id = recommend_ids[i]
            p_name = place_df.iloc[p_id,1]
            metadata = {"placeID": int(p_id), "Name": p_name}
            cb_recommend.append(metadata)



        response_ = {"Content Based NLP": cb_recommend}
        return jsonify(response_)

class FlaskAPI(object):
    def __init__(self):

        run_event = threading.Event()
        run_event.set()
        # threading.Thread(target=self.recommendation_system, args = [restaurants,rating_pivot, knn_user_based, run_event]).start()
        app.run(port=5000, host='0.0.0.0')
        try:
            while 1:
                sleep(.1)
        except KeyboardInterrupt:
            print("Closing server...")
            run_event.clear()



FlaskAPI()