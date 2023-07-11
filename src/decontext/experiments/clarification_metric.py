"""Calculate the Clarification (CLF) metric"""

import copy
import re
from collections import defaultdict
from typing import Any, Optional

import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stopwords = stopwords.words("english")
punctuation = "()[]{},.?!/''\"``"
ps = PorterStemmer()


def get_keywords(sentence: str) -> set[str]:
    """Extract keywords from a sentence.

    Extract keywords from a sentence by lowercasing, tokenizing and stemming words. Punctuation and
    stopwords ar filtered out. Only unique words are returned.

    Args:
        sentence (str): The text to extract keywords from.

    Returns:
        set[str] containing the keywords in the sentence.
    """

    return set(
        [
            ps.stem(word.lower())
            for word in word_tokenize(sentence)
            if word not in stopwords and word not in punctuation
        ]
    )


def extract_additions(decontextualization: str) -> list[str]:
    """Extract the additions from decontextualized snippets.

    The text that is added to decontextualized snippets should be put between square brackets "[]".
    Extract all of this text.

    Args:
        decontextualization (str): The decontextualized snippet.

    Returns:
        list[str] containing the text that was added between brackets to the decontextualization.
    """

    pattern = r"\[(.*?)\]"
    target_additions = re.findall(pattern, decontextualization)
    return target_additions


def jaccard(a: set[Any], b: set[Any]) -> float:
    """Calculate and return the Jaccard similarity between set `a` and `b`."""

    return len(a & b) / len(a | b)


def jaccard_sentence(
    sentence_1: str, sentence_2: str, keywords: bool = True
) -> float:
    """Calculate the Jaccard similarity between sentences.

    If `keywords` is True, only calculate the similarity using the keywords. If False, then use all of the tokens
    as tokenized by nltk.word_tokenize.

    Args:
        sentence_1 (str): The first sentence.
        sentence_2 (str): The second sentence.
        keywords (bool): Whether to use keywords (True) or all tokens (False). Default is True.

    Returns:
        The Jaccard similarity between the two sentences (float).
    """

    if keywords:
        sentence_1_tokens = get_keywords(sentence_1)
        sentence_2_tokens = get_keywords(sentence_2)
    else:
        sentence_1_tokens = set(word_tokenize(sentence_1))
        sentence_2_tokens = set(word_tokenize(sentence_2))
    return jaccard(sentence_1_tokens, sentence_2_tokens)


def get_p_r_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Get the precision, recall, and F1 given true positives, false positives, and false negatives.

    Args:
        tp (int): True positives.
        fp (int): False positives.
        fn (int): False negatives.

    Returns:
        Tuple[float, float, float] containing (precision, recall, F1).
    """

    if tp == 0:
        return 0, 0, 0
    else:
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        return p, r, 2 * p * r / (p + r)


def compute_clf_metric(
    prediction_additions: list[Optional[str]],
    target_additions: list[Optional[str]],
    debug: bool = False,
) -> tuple[float, float, float, float, list[dict]]:
    """Compute the CLF metric.

    First, align each predicted addition to its closest target additions using jaccard similarity. Then calculate
    the precision, recall, and F1 of the aligned vs unaligned additions.

    Args:
        prediction_additions (list[str]): Spans of text that were added to the predictions.
        target_additions (list[str]): Spans of text that were added to the gold data.
        debug (bool): Whether to print intermediate debugging information (True) or not (False). Default False.

    Returns:
        Tuple containing five elements:
            * precision
            * recall
            * f1
            * mean Jaccard similarity of the closest prediction and target additions
            * list of the closest alignments between prediction and target additions along with
              the jaccard similarity between them.
    """

    prediction_additions = copy.deepcopy(prediction_additions)
    target_additions = copy.deepcopy(target_additions)
    prediction_tokens: list[Optional[set[str]]] = [
        set(
            [
                ps.stem(word)
                for word in word_tokenize(addition)
                if word not in stopwords and word not in punctuation
            ]
        )
        for addition in prediction_additions
    ]
    target_tokens = [
        set(
            [
                ps.stem(word)
                for word in word_tokenize(addition)
                if word not in stopwords and word not in punctuation
            ]
        )
        for addition in target_additions
    ]

    # calculate overlaps
    closest_target_idxs: list[Optional[int]] = []
    scores = []
    for pred_tokens in prediction_tokens:
        # This shouldn't be true, but it's to satisfy mypy
        if pred_tokens is None:
            continue
        pred_ious = []
        for targ_tokens in target_tokens:
            pred_ious.append(
                len(pred_tokens & targ_tokens) / len(pred_tokens)
                if pred_tokens
                else 0
            )

        if pred_ious:
            closest_target_idxs.append(np.argmax(pred_ious))
        scores.append(pred_ious)

    # if two predictions match the same target, merge the predictions
    snippet_match_threshold = 0.25
    scores = np.array(scores)
    pred_merge_dict = defaultdict(list)
    for pred_idx, targ_idx in enumerate(closest_target_idxs):
        # This should never run, but it's to satisfy mypy
        if pred_idx is None or targ_idx is None:
            continue

        if debug:
            print(
                prediction_additions[pred_idx],
                "->",
                target_additions[targ_idx],
                f"({scores[pred_idx, targ_idx]:.3f})",
            )
        if scores[pred_idx, targ_idx] > snippet_match_threshold:
            pred_merge_dict[targ_idx].append(pred_idx)

    for targ_idx, pred_idxs_to_merge in pred_merge_dict.items():
        if len(pred_idxs_to_merge) > 1:
            # merge the two predictions - have to update
            # `closest_target_idxs`, `prediction_tokens`, `prediction_additions`
            first_pred_idx = pred_idxs_to_merge[0]

            # Extra verbosity here is to satisfy mypy
            _new_additions: Optional[str] = prediction_additions[
                first_pred_idx
            ]
            new_additions: str = ""
            if _new_additions is not None:
                new_additions = _new_additions

            new_tokens = prediction_tokens[first_pred_idx]
            for idx in pred_idxs_to_merge[
                -1:0:-1
            ]:  # Go from the end in reverse, stopping before the first element

                # Extra verbosity here is to satisfy mypy
                _next_addition: Optional[str] = prediction_additions[idx]
                next_addition: str = ""
                if _next_addition is not None:
                    next_addition = _next_addition
                try:
                    new_additions += " | " + next_addition
                    prediction_additions[idx] = None
                except IndexError:
                    print(pred_idxs_to_merge)
                    print(prediction_additions)
                    raise IndexError

                # Extra verbosity is to please mypy
                _new_pred_tokens = prediction_tokens[idx]
                new_pred_tokens = set()
                if _new_pred_tokens is not None:
                    new_pred_tokens = _new_pred_tokens
                if new_tokens is not None:
                    new_tokens |= new_pred_tokens
                prediction_tokens[idx] = None
                closest_target_idxs[idx] = None

            prediction_additions[first_pred_idx] = new_additions
            prediction_tokens[first_pred_idx] = new_tokens

    prediction_additions = [
        addition for addition in prediction_additions if addition is not None
    ]
    prediction_tokens = [
        tokens for tokens in prediction_tokens if tokens is not None
    ]
    closest_target_idxs = [
        idxs for idxs in closest_target_idxs if idxs is not None
    ]

    if debug:
        print()
        print("After merging:")
        print("Prediction additions:", prediction_additions)
        print("Target additions:", target_additions)
        print()

    # calculate snippet-level precision/recall
    tp = 0
    fp = 0
    fn = len(target_additions) - len(
        set(closest_target_idxs)
    )  # how many target snippets are missed?
    for pred_idx, targ_idx in enumerate(closest_target_idxs):
        if scores[pred_idx, targ_idx] > snippet_match_threshold:
            tp += 1
        else:
            fp += 1

    p, r, f1 = get_p_r_f1(tp, fp, fn)
    if debug:
        print("Added span-level p/r/f1:")
        print(f"tp: {tp}, fp: {fp}, fn: {fn} | P: {p} R: {r} F1: {f1}")
        print()

    # Now, of the snippets that match, how well do they match?
    # We want to report the IoU for the prediction vs target snippets that match:
    # This is different from the score we previously calculated, (where the denominator was
    # len(prediction) ), but looks similar

    iou_scores = []
    alignments = []
    for pred_idx, targ_idx in enumerate(closest_target_idxs):
        if scores[pred_idx, targ_idx] > snippet_match_threshold:
            pred_tokens = prediction_tokens[pred_idx]
            targ_tokens = target_tokens[targ_idx]
            iou_score = jaccard(pred_tokens, targ_tokens)
            iou_scores.append(iou_score)
            alignments.append(
                {
                    "prediction_addition": prediction_additions[pred_idx],
                    "target_addition": target_additions[targ_idx],
                    "iou_score": float(iou_score),
                    "included_in_iou": True,
                }
            )
            if debug:
                print(
                    prediction_additions[pred_idx],
                    "->",
                    target_additions[targ_idx],
                    f"IOU: {iou_scores[-1]}",
                )
        else:
            alignments.append(
                {
                    "prediction_addition": prediction_additions[pred_idx],
                    "target_addition": target_additions[targ_idx],
                    "match_score": float(scores[pred_idx, targ_idx]),
                    "included_in_iou": False,
                }
            )

    for targ_idx, target_addition in enumerate(target_additions):
        if targ_idx not in closest_target_idxs:
            alignments.append(
                {
                    "prediction_additions": None,
                    "target_addition": target_addition,
                    "included_in_iou": False,
                }
            )

    if debug:
        print("Of the additions that match, calculate iou")
        print(iou_scores, np.mean(iou_scores))

    return (
        float(p),
        float(r),
        float(f1),
        float(np.mean(iou_scores)),
        alignments,
    )
