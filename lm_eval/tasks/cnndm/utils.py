
import numpy as np
from rouge_score import rouge_scorer, scoring

# code copied from truthfulqa evaluation

def process_results_gen(doc, results):
    completion = results[0]
    ref = doc["highlights"]

    # Process the sentence-level BLEURT, BLEU, and ROUGE for similarity measures.

    # ROUGE-N
    rouge_scores = rouge([ref], [completion]) 
    # ROUGE-1
    rouge1_score = rouge_scores["rouge1"]
    # ROUGE-2
    rouge2_score = rouge_scores["rouge2"]
    # ROUGE-L
    rougeL_score = rouge_scores["rougeLsum"]

    return {
        "rouge_1": rouge1_score,
        "rouge_2": rouge2_score,
        "rouge_l": rougeL_score
    }

def rouge(refs, preds):
    """
    Returns `t5` style ROUGE scores. See the related implementation:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/3d10afd51ba97ac29eb66ae701eca274488202f7/t5/evaluation/metrics.py#L68

    :param refs:
        A `list` of reference `strs`.
    :param preds:
        A `list` of predicted `strs`.
    """
    rouge_types = ["rouge1", "rouge2", "rougeLsum"]
    scorer = rouge_scorer.RougeScorer(rouge_types)
    # Add newlines between sentences to correctly compute `rougeLsum`.

    def _prepare_summary(summary):
        summary = summary.replace(" . ", ".\n")
        return summary

    # Accumulate confidence intervals.
    aggregator = scoring.BootstrapAggregator()
    for ref, pred in zip(refs, preds):
        ref = _prepare_summary(ref)
        pred = _prepare_summary(pred)
        aggregator.add_scores(scorer.score(ref, pred))
    result = aggregator.aggregate()
    return {type: result[type].mid.fmeasure * 100 for type in rouge_types}