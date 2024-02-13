# LeQua 2024: Learning to Quantify

This repository contains the official [evaluation script](evaluate.py) that will be used
for evaluating submissions to the LeQua2024 competition. 
This repository also provides a [format checker](format_checker.py),
in order to allow participants to check that the format of their submissions
is correct.
Additionally, some helper methods are made available for the convenience of participants.

## What is LeQua 2024?

LeQua2024 is the 2nd data challenge on Learning to Quantify.
The aim of this challenge is to allow the comparative evaluation 
of methods for “learning to quantify”, i.e., methods
for training predictors of the relative frequencies of the 
classes of interest in sets of unlabelled documents.
For further details, please visit [the official LeQua2024's site](https://lequa2024.github.io/).

## Datasets

The datasets of the challenge [will be published on Zenodo](https://www.doi.org/).

There are four [tasks](https://lequa2024.github.io/tasks/), each with its own data:
- T1: binary quantification.
- T2: multiclass quantification.
- T3: ordinal quantification.
- T4: binary quantification under covariate shift.

Data for every task is published in three parts (`Tn` is used to denote any task):
- `Tn.train_dev.zip`: released on Feb 15th, 2024, that contains the training and development data.
- `Tn.test.zip`: released on May 1st, 2024, that contains the test data, without true prevalence values, for which the participants have to produce their submissons.
- `Tn.test_prevalences.zip`: released after the end of the challenge, that contains the true prevalence values for the test data.

The `Tn.train_dev.zip` file contains:
- `Tn/public/training_data.txt`: lists the training documents (as pre-processed numerical vectors, so as to focus on quantification and not on text processing) with labels.
  All files are `comma separated values` files in `.txt` format, with a header row, and can be loaded, for example, with the Pandas command `df = pandas.read_csv(filename)`.
  
  Training data consists of 257 columns. The first column, called `label`, contains a zero-based numerical id for the label. The remaining 256 columns, labeled for `0` to `255` are the 256 dimensions of the vectors that represent the content of the documents.
  
- `Tn/public/label_map.txt`: pairs numerical id of labels with a human-readable name.

- `Tn/public/dev_samples`: directory containing the samples that can be used to develop the quantification methods. Every file in the directory is a sample for which to predict prevalence values of labels. The filename of a sample is a natural number that identifies the sample. The format of the files is the same one of the `training_data.txt` file, except that there is no `label` column.

- `Tn/public/dev_prevalences.txt`: lists the true prevalence values for the development samples. Every row corresponds to a different sample; the `id` column identifies the sample (e.g., id 3 indicates that the prevalence values correspond to sample `3.txt` in the `dev_samples` directory). Columns report the prevalence values for all the labels.

- `Tn/public/dummy_dev_predictions.txt`: an example file that uses the same format of submission files, yet referring to the development samples. A prediction file made following this format, together with the `dev_prevalences.txt` file, can be passed as the argument of the evaluation script (described below) to evaluate a quantification method on the development data.

The `Tn.test.zip` file contains:
  
- `Tn/public/test_samples`: directory containing the test samples for which prevalence prediction must be made. Files in the directory have the same format of those in the `dev_samples` directory.

- `Tn/public/dummy_test_predictions.txt`: shows the format of a submission file for the challenge. Use the format checker (described below) to check the format of your submission file.

The `Tn.test_prevalences.zip` contains the true prevalences for the test data.

## The evaluation script

The [evaluation script](evaluate.py) takes two result files, one containing
the true prevalence values (the ground truth) and another containing the estimated prevalence
values (a submission file), and computes the estimation error (in terms of the `mean relative absolute error` and
`mean absolute error` measures). The script can be run from the command line as follows (use
`--help` to display information on its use):

```
usage: evaluate.py [-h] [--output SCORES-PATH]
                   TASK TRUE-PREV-PATH PRED-PREV-PATH

LeQua2024 official evaluation script

positional arguments:
  TASK                  Task name (T1, T2, T3, T4)
  TRUE-PREV-PATH        Path of ground truth prevalence values file (.csv)
  PRED-PREV-PATH        Path of predicted prevalence values file (.csv)

optional arguments:
  -h, --help            show this help message and exit
  --output SCORES-PATH  Path where to store the evaluation scores
```

The error valuess are displayed in standard output and, optionally, dumped on a txt file.
For example:

> python3 evaluate.py T1A ./data/T1/public/dev_prevalences.txt ./my_submission/estimated_prevalences.txt --output scores.txt

*Note* that the first file corresponds to the ground truth prevalence values, and the second file
corresponds to the estimated prevalence values. The order is **not** interchangeable since 
relative absolute error is not symmetric.

## The format checker

The [format checker](format_checker.py) serves the purpose of checking
that the participants' submission files contain no formatting errors.
See the usage information (by typing `--help`):

```
usage: format_checker.py [-h] PREVALENCEFILE-PATH

LeQua2024 official format-checker script

positional arguments:
  PREVALENCEFILE-PATH  Path to the file containing prevalence values to check

optional arguments:
  -h, --help           show this help message and exit
```

Some mock submission files are provided as examples. For example, running

> python3 format_checker.py ./data/T1/public/dummy_dev_predictions.txt

will produce the output

> Format check: [passed]

If the format is not correct, the check will not be successful, and the checker will
display some hint regarding the type of error encountered.

## The submission files format:

Submission files will consist of `comma separated values` files with `txt` extension.
The format should comply with the following conditions:
* The header is: `id, 0, 1, <...>, n-1` where `n` is the number of classes 
  (`n=2` in tasks T1 and T4, `n=5` for task T3, and `n=28` in task T2)
* Every row consists of a document id (an integer) followed by the
  class prevalence values for each class. E.g., a valid row could be 
  `5,0.7,0.3`, meaning that the sample with id 5 (i.e., the file `5.txt`) has prevalence values of 0.7 for the class
  0 and 0.3 for the class 1.
* There should be exactly 5000 rows 
  (regardless of the task), after the header. 
  (Do not confuse with the development result files, that have only 1000 rows, and are nontheless not meant to be submitted.)
* Document ids must not contain gaps, and should start by 0.
* Prevalence values should be in the range [0,1] for all classes, and sum
  up to 1 (with an error tolerance of 1E-3).

The easiest way to create a valid submission file is by means of the
class `ResultSubmission` defined in [data.py](data.py). See the following 
section in which some useful functions are discussed.

Result files should be submitted to CodaLab (further details TBA). 

## Utilities

Some useful functions for managing the data are made available in
[data.py](data.py) for your convenience. The following example illustrates
some of them.

Before getting started, let's simulate the outputs of any quantifier 
by means of some mock predictions: 

```
from data import *
import numpy as np

def mock_prediction(n_classes):
    predicted_prevalence = np.random.rand(n_classes)
    predicted_prevalence /= predicted_prevalence.sum()
    return predicted_prevalence
```

This first example shows how to create a valid submission file for task T1.
Let us assume we already have a trained quantifier (in this example, one that
always returns mock predictions), 
and we want to assess
its performance on the development set (for which the true prevalence
values are known). 
Assume the development samples are located in `./data/T1/public/dev_samples`  
and the true prevalence are in `./data/T1/public/dev_prevalences.txt`.
This could be carried out as follows:

```
# create a new submission file
submission_T1 = ResultSubmission()

path_dir = './data/T1/public/dev_samples'
ground_truth_path = './data/T1/public/dev_prevalences.txt'

# iterate over devel samples
# the function that loads samples in vector form is "load_vector_documents"
for id, sample, prev in gen_load_samples(path_dir, ground_truth_path, return_id=True, load_fn=load_vector_documents):
    # replace this with the predictions generated by your quantifier
    predicted_prevalence = mock_prediction(n_classes=2)
    
    # append the predictions to the submission file
    submission_T1.add(id, predicted_prevalence)
    
# dump to file
submission_T1.dump('mock_submission.T1.dev.txt')
```

Participants can now use the [evaluation script](evaluate.py) to 
evaluate the quality of their predictions with respect to the ground truth.

After the test set is released (see the [timeline](https://lequa2024.github.io/timeline/)
for further details), participants will be asked
to generate predictions for the test samples, and to submit a prediction
file. Note that the ground truth prevalence values of the test samples
will not be made available until the competition finishes. 
The following script illustrates how to iterate over test samples
and generate a valid submission file.

```
submission_T1 = ResultSubmission()
path_dir = './data/T1/public/test_samples'

# the iterator has no access to the true prevalence values, since a
# file containing the ground truth values is not indicated 
for id, sample in gen_load_samples(path_dir, load_fn=load_vector_documents):
    submission_T1.add(id, mock_prediction(n_classes=2))
submission_T1.dump('mock_submission.T1.test.txt')
```

Note that only training documents are labelled. Development samples are (and test samples will be)
unlabelled, although the same function can be used to read both labelled and unlabelled data samples, 
as it automatically handles the presence or absence of the labels.

## QuaPy

A number of baseline (and advanced) methods for learning to quantify 
are implemented in the [QuaPy](https://github.com/HLT-ISTI/QuaPy/tree/lequa2024) Python-based, open-source library, 
which also contains implementations of standard evaluation 
measures and evaluation protocols. 
For participating in this lab you are welcome to use [QuaPy](https://github.com/HLT-ISTI/QuaPy/tree/lequa2024) and 
its tools in any way you might deem suitable (it is not mandatory, though).

All the official baselines for LeQua2024 will be implemented as part of QuaPy.
Check out the branch [lequa2024](https://github.com/HLT-ISTI/QuaPy/tree/lequa2024) in which
the baseline methods are implemented, and in which you can find many useful
tools for preparing, running, and evaluating your own experiments.

Check paper [Alejandro Moreo, Andrea Esuli, and Fabrizio Sebastiani. QuaPy: A Python-based framework for quantification. Proceedings of the 30th ACM International Conference on Knowledge Management (CIKM 2021), Gold Coast, AU, pp. 4534–4543](https://dl.acm.org/doi/10.1145/3459637.3482015) 
to learn more about QuaPy.
