"""
NOTE: We've curated a set of query-document relevance scores for you to use in this part of the assignment. 
You can find 'relevance.test.csv', where the 'rel' column contains scores of the following relevance levels: 
1 (non-relevant) to 5 (very relevant). When you calculate MAP, treat 4s and 5s as relevant documents. 
Treat search results from your ranking function that are not listed in the file as non-relevant.
"""
import csv
from tqdm import tqdm
import numpy as np
import math
import pandas as pd



def map_score(search_result_relevances: list[int], cut_off: int = 10) -> float:
    """
    Calculates the mean average precision score given a list of labeled search results, where
    each item in the list corresponds to a document that was retrieved and is rated as 0 or 1
    for whether it was relevant.

    Args:
        search_result_relevances: A list of 0/1 values for whether each search result returned by your
            ranking function is relevant
        cut_off: The search result rank to stop calculating MAP.
            The default cut-off is 10; calculate MAP@10 to score your ranking function.

    Returns:
        The MAP score
    """
    total_relevant = sum(search_result_relevances)

    if total_relevant == 0:
        return 0.0
    
    relevant_found = 0
    precision_sum = 0.0

    for idx, rel in enumerate(search_result_relevances[:cut_off]):
        if rel == 1:
            relevant_found += 1
            precision_sum += relevant_found / (idx + 1)

    return precision_sum / min(total_relevant, cut_off)


def ndcg_score(search_result_relevances: list[float],
               ideal_relevance_score_ordering: list[float], cut_off: int = 10):
    """
    Calculates the normalized discounted cumulative gain (NDCG) given a lists of relevance scores.
    Relevance scores can be ints or floats, depending on how the data was labeled for relevance.
    Use the lecture definition DCG@k = sum(rel_i / log2(i + 1)) for one-based
    rank i. With zero-based Python indexing, the denominator is log2(i + 2).

    Args:
        search_result_relevances: A list of relevance scores for the results returned by your ranking function
            in the order in which they were returned
            These are the human-derived document relevance scores, *not* the model generated scores.
        ideal_relevance_score_ordering: The list of relevance scores for results for a query, sorted by relevance score
            in descending order
            Use this list to calculate IDCG (Ideal DCG).

        cut_off: The default cut-off is 10.

    Returns:
        The NDCG score
    """
    dcg = 0
    idcg = 0

    for idx in range(cut_off):
        dcg += search_result_relevances[idx] / math.log2(idx + 2)
        idcg += ideal_relevance_score_ordering[idx] / math.log2(idx + 2)

    if idcg == 0:
        return 0

    ndcg = dcg / idcg

    return ndcg


def run_relevance_tests(relevance_data_filename: str, ranker) -> dict[str, float]:
    """
    Measures the performance of the IR system using metrics, such as MAP and NDCG.

    Args:
        relevance_data_filename: The filename containing the relevance data to be loaded
        ranker: A ranker configured with a particular scoring function to search through the document collection.
            This is probably either a Ranker or a L2RRanker object, but something that has a query() method.

    Returns:
        A dictionary containing both MAP and NDCG scores
    """
    # Load dataset
    df = pd.read_csv(relevance_data_filename, encoding='latin-1')

    # Construct MAP relevance score
    df["map_rel"] = df["rel"].apply(lambda x: 1 if x > 3 else 0)

    # Get list of queries
    queries = df["query"].drop_duplicates()

    map_list = []
    ndcg_list = []

    for query in queries:

        query_df = df[df["query"] == query]

        results = ranker.query(query)
        results = pd.DataFrame(results, columns=["docid", "ranker_rel"])

        results = results.merge(query_df, on=["docid"], how="left")
        results["rel"] = results["rel"].fillna(0)
        results["map_rel"] = results["map_rel"].fillna(0)

        # results = results.sort_values("ranker_rel", ascending=False)

        mscore = map_score(results["map_rel"].tolist())

        ideal_relevance = query_df["rel"].sort_values(ascending=False).tolist()
        nscore = ndcg_score(results["rel"].tolist(), ideal_relevance)

        map_list.append(mscore)
        ndcg_list.append(nscore)

    return {'map': np.mean(map_list), 'ndcg': np.mean(ndcg_list), 'map_list': map_list, 'ndcg_list': ndcg_list}


if __name__ == '__main__':
    pass
