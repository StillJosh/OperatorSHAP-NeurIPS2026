"""Utilities for PDE data generation."""

import contextlib

import joblib


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Route joblib.Parallel progress into a tqdm progress bar.

    Standard joblib snippet: swap in a BatchCompletionCallBack subclass that
    increments the given tqdm bar, restore the original on exit.

    Parameters
    ----------
    tqdm_object : tqdm.tqdm
        An already-constructed tqdm bar (with total=n set by the caller).

    Yields
    ------
    tqdm.tqdm
        The same bar, for use inside a ``with`` block around Parallel(...).
    """
    class _TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    original = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = original
        tqdm_object.close()
