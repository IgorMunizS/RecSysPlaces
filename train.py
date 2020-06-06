from utils import readDatabase
from sklearn.model_selection import train_test_split
from models import PopularityRecommender, CFR, ContentBasedNLP, HybridRecommender,ContentBasedNLPHotels
from models import ContentBasedNLPMuseums
from evaluation import Evaluator

class Train:

    def __init__(self):
        self.place_df, self.user_df, self.hotels_df, self.museums_df = readDatabase()

        self.user_train_df, self.user_test_df = train_test_split(self.user_df,
                                                       stratify=self.user_df['userID'],
                                                       test_size=0.20,
                                                       random_state=42)

        self.train_and_evaluate()
        self.train_full_data_and_save()

    def train_and_evaluate(self):
        model_evaluator = Evaluator(self.place_df, self.user_df, self.user_train_df, self.user_test_df)

        print('Training Popularity recommendation model...')
        popularity_model = PopularityRecommender(self.place_df)
        popularity_model.train_model(self.user_train_df)
        print('Evaluating Popularity recommendation model...')
        pop_global_metrics, pop_detailed_results_df = model_evaluator.evaluate_model(popularity_model)
        print('\n Metrics:\n%s' % pop_global_metrics)

        print('Training Collaborative Filtering recommendation model...')
        cfr_model = CFR(self.place_df)
        cfr_model.train_model(self.user_train_df)
        print('Evaluating Collaborative Filtering recommendation model...')
        pop_global_metrics, pop_detailed_results_df = model_evaluator.evaluate_model(cfr_model)
        print('\n Metrics:\n%s' % pop_global_metrics)

        print('Training Content Based NLP recommendation model...')
        cb_model = ContentBasedNLP(self.place_df)
        cb_model.train_model(self.user_train_df)
        print('Evaluating Content Based NLP recommendation model...')
        pop_global_metrics, pop_detailed_results_df = model_evaluator.evaluate_model(cb_model)
        print('\n Metrics:\n%s' % pop_global_metrics)

        print('Evaluating Hybrid recommendation model...')

        hybrid_model = HybridRecommender(cb_model, cfr_model, self.place_df,
                                                     cb_ensemble_weight=10, cf_ensemble_weight=100.0)

        pop_global_metrics, pop_detailed_results_df = model_evaluator.evaluate_model(hybrid_model)
        print('\n Metrics:\n%s' % pop_global_metrics)


    def train_full_data_and_save(self):
        print('Training Popularity recommendation model in full Dataset')
        popularity_model = PopularityRecommender(self.place_df)
        popularity_model.train_model(self.user_df)
        print('Saving Popularity recommendation model')
        popularity_model.save_model()

        print('Training Collaborative Filtering recommendation model...')
        cfr_model = CFR(self.place_df)
        cfr_model.train_model(self.user_df)
        print('Saving Collaborative Filtering recommendation model')
        cfr_model.save_model()

        print('Training Content Based NLP recommendation model...')
        cb_model = ContentBasedNLP(self.place_df)
        cb_model.train_model(self.user_df)
        print('Saving Content Based NLP recommendation model')
        cb_model.save_model()

        print('Training Content Based NLP Hotels recommendation model...')
        cb_model = ContentBasedNLPHotels(self.hotels_df)
        cb_model.train_model()
        print('Saving Content Based NLP Hotels recommendation model')
        cb_model.save_model()

        print('Training Content Based NLP Museums recommendation model...')
        cb_model = ContentBasedNLPMuseums(self.museums_df)
        cb_model.train_model()
        print('Saving Content Based NLP Museums recommendation model')
        cb_model.save_model()

train = Train()
