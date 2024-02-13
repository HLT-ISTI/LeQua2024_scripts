import os.path
from typing import Tuple, Union
import pandas as pd
import numpy as np
from glob import glob

import constants


def load_category_map(path):
    """
    Loads the category map, i.e., a mapping of numerical ids of labels with a human readable name.

    :param path: path to the label map file
    :return: a dictionary cat2code (i.e., cat2code[cat_name] gives access to the category id) and a list code2cat (i.e.,
        code2cat[cat_id] gives access to the category name)
    """
    cat2code = {}
    with open(path, 'rt') as fin:
        for line in fin:
            category, code = line.split()
            cat2code[category] = int(code)
    code2cat = [cat for cat, code in sorted(cat2code.items(), key=lambda x:x[1])]
    return cat2code, code2cat


def load_vector_documents(path):
    """
    Loads vectorized documents. In case the sample is unlabelled,
    the labels returned are None

    :param path: path to the data sample containing the raw documents
    :return: a tuple with the documents (np.ndarray of shape `(n,256)`) and the labels (a np.ndarray of shape `(n,)` if
        the sample is labelled, or None if the sample is unlabelled), with `n` the number of instances in the sample
        (250 for T1 and T4, 1000 for T2, and 200 for T3)
    """
    D = pd.read_csv(path).to_numpy(dtype=float)
    labelled = D.shape[1] == 257
    if labelled:
        X, y = D[:,1:], D[:,0].astype(int).flatten()
    else:
        X, y = D, None
    return X, y


def __gen_load_samples_with_groudtruth(path_dir:str, return_id:bool, ground_truth_path:str):
    true_prevs = ResultSubmission.load(ground_truth_path)
    for id, prevalence in true_prevs.iterrows():
        sample, _ = load_vector_documents(os.path.join(path_dir, f'{id}.txt'))
        yield (id, sample, prevalence) if return_id else (sample, prevalence)


def __gen_load_samples_without_groudtruth(path_dir:str, return_id:bool):
    nsamples = len(glob(os.path.join(path_dir, f'*.txt')))
    for id in range(nsamples):
        sample, _ = load_vector_documents(os.path.join(path_dir, f'{id}.txt'))
        yield (id, sample) if return_id else sample


def gen_load_samples(path_dir:str, ground_truth_path:str = None, return_id=True):
    """
    A generator that iterates over samples (for which the prevalence values are either known or unknown). In case
    the file containing the ground truth prevalence values is indicated, the iterator returns the prevalence of the
    sample.

    :param path_dir: path to the folder containing the samples
    :param ground_truth_path: if indicated, points to the file of ground truth prevalence values for each sample
    :param return_id: set to True (default) to return the sample id
    :return: each iteration consists of a tuple containing the id of the sample (if `return_id=True`), the data sample
        (in any case), and the prevalence values (if `ground_truth_path` has been specified)
    """
    if ground_truth_path is None:
        # the generator function returns tuples (docid:str, sample:np.ndarray)
        gen_fn = __gen_load_samples_without_groudtruth(path_dir, return_id)
    else:
        # the generator function returns tuples (docid:str, sample:np.ndarray)
        gen_fn = __gen_load_samples_with_groudtruth(path_dir, return_id, ground_truth_path)
    for r in gen_fn:
        yield r


class ResultSubmission:
    """
    A container for the submission results. This class implements routines to load, dump, and create iteratively
    a valid result file.
    """

    def __init__(self):
        self.df = None

    def __init_df(self, categories:int):
        if not isinstance(categories, int) or categories < 2:
            raise TypeError('wrong format for categories: an int (>=2) was expected')
        df = pd.DataFrame(columns=list(range(categories)))
        df.index.set_names('id', inplace=True)
        self.df = df

    @property
    def n_categories(self):
        """
        Returns the number of categories of this result container

        :return: the number of categories
        """
        return len(self.df.columns.values)

    def add(self, sample_id:int, prevalence_values:np.ndarray):
        """
        Adds a prevalence value vector for a sample.

        :param sample_id: the id of the sample (int)
        :param prevalence_values: the prevalence values for each class (np.ndarray)
        """
        if not isinstance(sample_id, int):
            raise TypeError(f'error: expected int for sample_sample, found {type(sample_id)}')
        if not isinstance(prevalence_values, np.ndarray):
            raise TypeError(f'error: expected np.ndarray for prevalence_values, found {type(prevalence_values)}')
        if self.df is None:
            self.__init_df(categories=len(prevalence_values))
        if sample_id in self.df.index.values:
            raise ValueError(f'error: prevalence values for "{sample_id}" already added')
        if prevalence_values.ndim != 1 and prevalence_values.size != self.n_categories:
            raise ValueError(f'error: wrong shape found for prevalence vector {prevalence_values}')
        if (prevalence_values < 0).any() or (prevalence_values > 1).any():
            raise ValueError(f'error: prevalence values out of range [0,1] for "{sample_id}"')
        if np.abs(prevalence_values.sum()-1) > constants.ERROR_TOL:
            raise ValueError(f'error: prevalence values do not sum up to one for "{sample_id}"'
                             f'(error tolerance {constants.ERROR_TOL})')

        self.df.loc[sample_id] = prevalence_values

    def __len__(self):
        """
        Number of samples for which the prevalence values are indicated

        :return: integer
        """
        return len(self.df)

    @classmethod
    def load(cls, path: str) -> 'ResultSubmission':
        """
        Loads a file containing the prevalence values of each data sample

        :param path: string
        :return: an instance of `ResultSubmission`
        """
        df = ResultSubmission.check_file_format(path)
        r = ResultSubmission()
        r.df = df
        return r

    def dump(self, path:str):
        """
        Dumps the information to a file.

        :param path: string
        """
        ResultSubmission.check_dataframe_format(self.df)
        self.df.to_csv(path)

    def prevalence(self, sample_id:int):
        """
        Returns the prevalence values indicated for a given sample

        :param sample_id: integer, the id of the sample
        :return: a np.ndarray if the sample with id `sample_id` exists in the container, or None otherwise
        """

        sel = self.df.loc[sample_id]
        if sel.empty:
            return None
        else:
            return sel.values.flatten()

    def iterrows(self):
        """
        An iterator over rows of the container

        :return: an iterator yielding the sample id (integer) and the prevalence (np.ndarray)
        """
        for index, row in self.df.iterrows():
            prevalence = row.values.flatten()
            yield index, prevalence

    @classmethod
    def check_file_format(cls, path) -> Union[pd.DataFrame, Tuple[pd.DataFrame, str]]:
        """
        Checks whether the file indicated has valid format. If the format is not correct, an exception is raised
        indicating the type of error encountered. If the file is correct, an instance of `ResultSubmission` is
        returned.

        :param path: string
        :return: the `ResultSubmission` if the format check passes (if otherwise, an exception is raised)
        """
        df = pd.read_csv(path, index_col=0)
        return ResultSubmission.check_dataframe_format(df, path=path)

    @classmethod
    def check_dataframe_format(cls, df, path=None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, str]]:
        """
        Implements the format check from the dataframe content.

        :param df: a pandas' dataframe
        :param path: if indicated, is used within the hints of the exceptions that may be raised due to format errors
        :return: the `ResultSubmission` if the format check passes (if otherwise, an exception is raised)
        """
        hint_path = ''  # if given, show the data path in the error message
        if path is not None:
            hint_path = f' in {path}'

        if df.index.name != 'id' or len(df.columns) < 2:
            raise ValueError(f'wrong header{hint_path}, '
                             f'the format of the header should be "id,0,...,n-1", '
                             f'where n is the number of categories')
        if [int(ci) for ci in df.columns.values] != list(range(len(df.columns))):
            raise ValueError(f'wrong header{hint_path}, category ids should be 0,1,2,...,n-1, '
                             f'where n is the number of categories')
        if df.empty:
            raise ValueError(f'error{hint_path}: results file is empty')
        elif len(df) != constants.DEV_SAMPLES and len(df) != constants.TEST_SAMPLES:
            raise ValueError(f'wrong number of prevalence values found{hint_path}; '
                             f'expected {constants.DEV_SAMPLES} for development sets and '
                             f'{constants.TEST_SAMPLES} for test sets; found {len(df)}')

        ids = set(df.index.values)
        expected_ids = set(range(len(df)))
        if ids != expected_ids:
            missing = expected_ids - ids
            if missing:
                raise ValueError(f'there are {len(missing)} missing ids{hint_path}: {sorted(missing)}')
            unexpected = ids - expected_ids
            if unexpected:
                raise ValueError(f'there are {len(missing)} unexpected ids{hint_path}: {sorted(unexpected)}')

        for category_id in df.columns:
            if (df[category_id] < 0).any() or (df[category_id] > 1).any():
                raise ValueError(f'error{hint_path} column "{category_id}" contains values out of range [0,1]')

        prevs = df.values
        round_errors = np.abs(prevs.sum(axis=-1) - 1.) > constants.ERROR_TOL
        if round_errors.any():
            raise ValueError(f'warning: prevalence values in rows with id {np.where(round_errors)[0].tolist()} '
                              f'do not sum up to 1 (error tolerance {constants.ERROR_TOL}), '
                              f'probably due to some rounding errors.')

        return df

