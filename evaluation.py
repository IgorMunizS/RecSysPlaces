import random
from utils import get_items_interacted
import pandas as pd

class Evaluator:

    def __init__(self, place_df,user_df,user_train_df,user_test_df,nb_non_items=10):
        self.nb_non_items = 10
        self.place_df = place_df

        self.user_indexed_df = user_df.set_index('userID')
        self.user_train_indexed_df = user_train_df.set_index('userID')
        self.user_test_indexed_df = user_test_df.set_index('userID')

    def get_not_interacted_items_sample(self, userID, sample_size, seed=42):
        interacted_items = get_items_interacted(userID, self.user_indexed_df)
        all_items = set(self.place_df['placeID'])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    def _verify_hit_top_n(self, item_id, recommended_items, topn):
        try:
            index = next(i for i, c in enumerate(recommended_items) if c == item_id)
        except:
            index = -1
        hit = int(index in range(0, topn))
        return hit, index

    def mean_reciprocal_rank(self, item, valid_pred):
        valid_pred = valid_pred.tolist()
        if item in valid_pred:
            index = valid_pred.index(item) + 1
            return (1 / index)
        else:
            return 0

    def evaluate_model_for_user(self, model, person_id):
        # Getting the items in test set
        interacted_values_testset = self.user_test_indexed_df.loc[person_id]
        if type(interacted_values_testset['placeID']) == pd.Series:
            person_interacted_items_testset = interacted_values_testset.sort_values('final_rating', ascending=False)[
                'placeID'].tolist()
        else:
            person_interacted_items_testset = [interacted_values_testset['placeID']]
        interacted_items_count_testset = len(person_interacted_items_testset)

        # Getting a ranked recommendation list from a model for a given user
        person_recs_df = model.recommend_items(person_id,
                                               items_to_ignore=get_items_interacted(person_id,
                                                                                    self.user_train_indexed_df),
                                               topn=10000000000)

        mrr = 0
        hits_at_5_count = 0
        hits_at_10_count = 0
        # For each item the user has interacted in test set
        for item_id in person_interacted_items_testset:
            # Getting a random sample (100) items the user has not interacted
            # (to represent items that are assumed to be no relevant to the user)
            non_interacted_items_sample = self.get_not_interacted_items_sample(person_id,
                                                                               sample_size=self.nb_non_items,
                                                                               seed=item_id % (2 ** 32))

            #             print(non_interacted_items_sample)

            # Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            #             print(items_to_filter_recs)

            # Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
            valid_recs_df = person_recs_df[person_recs_df['placeID'].isin(items_to_filter_recs)]
            valid_recs = valid_recs_df['placeID'].values
            mrr += self.mean_reciprocal_rank(item_id, valid_recs)
            # Verifying if the current interacted item is among the Top-N recommended items
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        # Recall is the rate of the interacted items that are ranked among the Top-N recommended items,
        # when mixed with a set of non-relevant items
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)

        mrr /= float(interacted_items_count_testset)

        person_metrics = {'hits@5_count': hits_at_5_count,
                          'hits@10_count': hits_at_10_count,
                          'interacted_count': interacted_items_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10,
                          'mean-reciprocal-rank': mrr}
        return person_metrics

    def evaluate_model(self, model):
        # print('Running evaluation for users')
        people_metrics = []
        for idx, person_id in enumerate(list(self.user_test_indexed_df.index.unique().values)):
            # if idx % 100 == 0 and idx > 0:
            #    print('%d users processed' % idx)
            person_metrics = self.evaluate_model_for_user(model, person_id)
            person_metrics['_person_id'] = person_id
            people_metrics.append(person_metrics)
        print('%d users processed' % idx)

        detailed_results_df = pd.DataFrame(people_metrics) \
            .sort_values('interacted_count', ascending=False)

        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(
            detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(
            detailed_results_df['interacted_count'].sum())
        global_mrr = detailed_results_df['mean-reciprocal-rank'].sum() / float(
            detailed_results_df['interacted_count'].sum())

        global_metrics = {'modelName': model.get_model_name(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10,
                          'mean-reciprocal-rank': global_mrr}
        return global_metrics, detailed_results_df