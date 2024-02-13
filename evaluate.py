import argparse
import numpy as np
import os
from pathlib import Path
from constants import LEQUA2024_TASKS, SAMPLE_SIZE
from data import ResultSubmission


"""
LeQua2024 Official evaluation script 
"""

def main(args):

    sample_size = SAMPLE_SIZE[args.task]

    true_prev = ResultSubmission.load(args.true_prevalences)
    pred_prev = ResultSubmission.load(args.pred_prevalences)

    mrae, mae = evaluate_submission(true_prev, pred_prev, sample_size)
    print(f'MRAE: {mrae:.4f}')
    print(f'MAE: {mae:.4f}')

    if args.output is not None:
        with open(args.output, 'at') as foo:
            foo.write(f'{args.pred_prevalences}\n')
            foo.write(f'MRAE: {mrae:.4f}\n')
            foo.write(f'MAE: {mae:.4f}\n\n')


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


def relative_absolute_error(p, p_hat, eps=None):
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

    p = __smooth(p, eps)
    p_hat = __smooth(p_hat, eps)
    return (abs(p-p_hat)/p).mean(axis=-1)


def evaluate_submission(true_prevs: ResultSubmission, predicted_prevs: ResultSubmission, sample_size, average=True):
    if len(true_prevs) != len(predicted_prevs):
        raise ValueError(f'size mismatch, ground truth file has {len(true_prevs)} entries '
                         f'while the file of predictions contains {len(predicted_prevs)} entries')
    if true_prevs.n_categories != predicted_prevs.n_categories:
        raise ValueError(f'these result files are not comparable since the categories are different: '
                         f'true={true_prevs.n_categories} categories vs. '
                         f'predictions={predicted_prevs.n_categories} categories')
    rae, ae = [], []
    for sample_id, true_prevalence in true_prevs.iterrows():
        pred_prevalence = predicted_prevs.prevalence(sample_id)
        rae.append(relative_absolute_error(true_prevalence, pred_prevalence, eps=1./(2*sample_size)))
        ae.append(absolute_error(true_prevalence, pred_prevalence))

    rae = np.asarray(rae)
    ae = np.asarray(ae)

    if average:
        rae = rae.mean()
        ae = ae.mean()

    return rae, ae


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LeQua2024 official evaluation script')
    parser.add_argument('task', metavar='TASK', type=str, choices=LEQUA2024_TASKS,
                        help='Task name (T1, T2, T3, T4)')
    parser.add_argument('true_prevalences', metavar='TRUE-PREV-PATH', type=str,
                        help='Path of ground truth prevalence values file (.csv)')
    parser.add_argument('pred_prevalences', metavar='PRED-PREV-PATH', type=str,
                        help='Path of predicted prevalence values file (.csv)')
    parser.add_argument('--output', metavar='SCORES-PATH', type=str, default=None,
                        help='Path where to store the evaluation scores')
    args = parser.parse_args()

    if args.output is not None:
        # ir specified, creates the path to the output file
        parentdir = Path(args.output).parent
        if parentdir:
            os.makedirs(parentdir, exist_ok=True)

    main(args)
