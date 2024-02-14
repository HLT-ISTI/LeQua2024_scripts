import argparse
import numpy as np
import os
from pathlib import Path
from constants import LEQUA2024_TASKS, SAMPLE_SIZE, VALID_MEASURES, ERROR_TOL
from data import ResultSubmission


"""
LeQua2024 Official evaluation script 
"""

def main(args):

    sample_size = SAMPLE_SIZE[args.task]

    true_prev = ResultSubmission.load(args.true_prevalences)
    pred_prev = ResultSubmission.load(args.pred_prevalences)

    print(f'{args.pred_prevalences}')

    if args.task == 'T3':
        # ordinal
        mnmd = evaluate_submission(true_prev, pred_prev, sample_size, measure='nmd', average=False)
        print(f'MNMD: {mnmd.mean():.5f} ~ {mnmd.std():.5f}')
    else:
        # non-ordinal
        mrae = evaluate_submission(true_prev, pred_prev, sample_size, measure='rae', average=False)
        mae  = evaluate_submission(true_prev, pred_prev, sample_size, measure='ae',  average=False)
        print(f'MRAE: {mrae.mean():.5f} ~ {mrae.std():.5f}')
        print(f'MAE: {mae.mean():.5f} ~ {mae.std():.5f}')

    print()


# -----------------------------------------------------------------------------------------------
# evaluation measures for T1, T2, T4 (the official one is relative_absolute_error)
# -----------------------------------------------------------------------------------------------

def absolute_error(prevs, prevs_hat):
    """Computes the absolute error between the two prevalence vectors.
     Absolute error between two prevalence vectors :math:`p` and :math:`\\hat{p}`  is computed as
     :math:`AE(p,\\hat{p})=\\frac{1}{|\\mathcal{Y}|}\\sum_{y\in \mathcal{Y}}|\\hat{p}(y)-p(y)|`,
     where :math:`\\mathcal{Y}` are the classes of interest.

    :param prevs: array-like of shape `(n_classes,)` with the true prevalence values
    :param prevs_hat: array-like of shape `(n_classes,)` with the predicted prevalence values
    :return: absolute error
    """
    assert prevs.shape == prevs_hat.shape, f'wrong shape {prevs.shape} vs. {prevs_hat.shape}'
    return abs(prevs_hat - prevs).mean(axis=-1)


def relative_absolute_error(prevs, prevs_hat, eps=None):
    """Computes the absolute relative error between the two prevalence vectors.
     Relative absolute error between two prevalence vectors :math:`p` and :math:`\\hat{p}`  is computed as
     :math:`RAE(p,\\hat{p})=\\frac{1}{|\\mathcal{Y}|}\\sum_{y\in \mathcal{Y}}\\frac{|\\hat{p}(y)-p(y)|}{p(y)}`,
     where :math:`\\mathcal{Y}` are the classes of interest.
     The distributions are smoothed using the `eps` factor (see :meth:`quapy.error.smooth`).

    :param prevs: array-like of shape `(n_classes,)` with the true prevalence values
    :param prevs_hat: array-like of shape `(n_classes,)` with the predicted prevalence values
    :param eps: smoothing factor. `rae` is not defined in cases in which the true distribution contains zeros; `eps`
        is typically set to be :math:`\\frac{1}{2T}`, with :math:`T` the sample size. If `eps=None`, the sample size
        will be taken from the environment variable `SAMPLE_SIZE` (which has thus to be set beforehand).
    :return: relative absolute error
    """

    def __smooth(prevs, eps):
        n_classes = prevs.shape[-1]
        return (prevs + eps) / (eps * n_classes + 1)

    prevs = __smooth(prevs, eps)
    prevs_hat = __smooth(prevs_hat, eps)
    return (abs(prevs - prevs_hat) / prevs).mean(axis=-1)


# -----------------------------------------------------------------------------------------------
# evaluation measures for T3
# -----------------------------------------------------------------------------------------------

def normalized_match_distance(prevs, prevs_hat):
    """
    Computes the Normalized Match Distance; which is the Normalized Distance multiplied by the factor
    `1/(n-1)` to guarantee the measure ranges between 0 (best prediction) and 1 (worst prediction).

    :param prevs: array-like of shape `(n_classes,)` with the true prevalence values
    :param prevs_hat: array-like of shape `(n_classes,)` with the predicted prevalence values
    :return: float in [0,1]
    """
    n = len(prevs)
    return (1./(n-1))*match_distance(prevs, prevs_hat)


def match_distance(prevs, prevs_hat):
    """
    Computes the Match Distance, under the assumption that the cost in mistaking class i with class i+1 is 1 in
    all cases.

    :param prevs: array-like of shape `(n_classes,)` with the true prevalence values
    :param prevs_hat: array-like of shape `(n_classes,)` with the predicted prevalence values
    :return: float
    """
    P = np.cumsum(prevs)
    P_hat = np.cumsum(prevs_hat)
    assert np.isclose(P_hat[-1], 1.0, rtol=ERROR_TOL), \
        'arg error in match_distance: the array does not represent a valid distribution'
    distances = np.abs(P-P_hat)
    return distances[:-1].sum()


def evaluate_submission(
        true_prevs: ResultSubmission,
        predicted_prevs: ResultSubmission,
        sample_size: int,
        measure: str,
        average=True):
    """
    Function used to evaluate a result submission file.

    :param true_prevs: ResultSubmission, true prevalence values (provided as part of the LeQua 2024 data)
    :param predicted_prevs: ResultSubmission, estimated prevalence values (computed by a participant's method)
    :param sample_size: int, number of instances per sample (depends on the task), see constants.SAMPLE_SIZE
    :param measure: str, either "rae", "ae" for tasks T1, T2, and T4, or "nmd" for T3
    :param average: bool, indicates whether the values have to be averaged before being returned
    :return: an array of error values if `average=False', or a single float if `average=True'
    """

    if len(true_prevs) != len(predicted_prevs):
        raise ValueError(f'size mismatch, ground truth file has {len(true_prevs)} entries '
                         f'while the file of predictions contains {len(predicted_prevs)} entries')
    if true_prevs.n_categories != predicted_prevs.n_categories:
        raise ValueError(f'these result files are not comparable since the categories are different: '
                         f'true={true_prevs.n_categories} categories vs. '
                         f'predictions={predicted_prevs.n_categories} categories')

    assert measure in VALID_MEASURES, f'unknown evaluation measure {measure}, valid ones are {VALID_MEASURES}'

    errors = []
    for sample_id, true_prevalence in true_prevs.iterrows():
        pred_prevalence = predicted_prevs.prevalence(sample_id)

        if measure == 'rae':
            err = relative_absolute_error(true_prevalence, pred_prevalence, eps=1./(2*sample_size))
        elif measure == 'ae':
            err = absolute_error(true_prevalence, pred_prevalence)
        elif measure == 'nmd': # for T3
            err = normalized_match_distance(true_prevalence, pred_prevalence)

        errors.append(err)

    errors = np.asarray(errors)

    if average:
        errors = errors.mean()

    return errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LeQua2024 official evaluation script')
    parser.add_argument('task', metavar='TASK', type=str, choices=LEQUA2024_TASKS,
                        help='Task name (T1, T2, T3, T4)')
    parser.add_argument('true_prevalences', metavar='TRUE-PREV-PATH', type=str,
                        help='Path of ground truth prevalence values file (.csv)')
    parser.add_argument('pred_prevalences', metavar='PRED-PREV-PATH', type=str,
                        help='Path of predicted prevalence values file (.csv)')
    args = parser.parse_args()

    main(args)
