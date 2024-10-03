import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from ..utils.utilities import EvalPrediction

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if not actual:
        return 0.0

    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        # first condition checks whether it is valid prediction
        # second condition checks if prediction is not repeated
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    return score / min(len(actual), k)

def compute_metrics(EvalPrediction:EvalPrediction):
    #Recall@k / MRR/ MAP@25
    # 
    metrics = {}
    metrics['AP@25'] = 0.0
    k = 25
    q_reps = EvalPrediction.predictions # bs,dim numpy
    p_reps = EvalPrediction.passages # bs,dim numpy
    y_true = EvalPrediction.label_ids # bs,1
    cosine_similarities  = cosine_similarity(q_reps,p_reps)
    sorted_indices = np.argsort(-cosine_similarities)[:,:k]
    for i in range(len(y_true)):
        actual_passages = [y_true[i]]
        predicted_passages = sorted_indices[i].tolist()
        metrics['AP@25'] += apk(actual_passages, predicted_passages, k=k)

    metrics['AP@25'] /= len(y_true)
    return metrics