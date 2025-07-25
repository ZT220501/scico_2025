# -*- coding: utf-8 -*-
# Copyright (C) 2020-2024 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""General utility functions."""


from __future__ import annotations

import numpy as np
import io
import socket
import urllib.error as urlerror
import urllib.request as urlrequest
from functools import reduce, wraps
from timeit import default_timer as timer
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import jax
from jax.interpreters.batching import BatchTracer
from jax.interpreters.partial_eval import DynamicJaxprTracer


def rgetattr(obj: object, name: str, default: Optional[Any] = None) -> Any:
    """Recursive version of :func:`getattr`.

    Args:
        obj: Object with the attribute to be accessed.
        name: Path to object in with components delimited by a "."
           character.
        default: Default value to be returned if the attribute does not
           exist.

    Returns:
        Attribute value of default if attribute does not exist.
    """

    try:
        return reduce(getattr, name.split("."), obj)
    except AttributeError as e:
        if default is not None:
            return default
        else:
            raise e


def rsetattr(obj: object, name: str, value: Any):
    """Recursive version of :func:`setattr`.

    Args:
        obj: Object with the attribute to be set.
        name: Path to object in with components delimited by a "."
           character.
        value: Value to which the attribute is to be set.
    """

    # See goo.gl/BVJ7MN
    path = name.split(".")
    setattr(reduce(getattr, path[:-1], obj), path[-1], value)


def partial(func: Callable, indices: Sequence, *fixargs: Any, **fixkwargs: Any) -> Callable:
    """Flexible partial function creation.

    This function is similar to :func:`functools.partial`, but allows
    fixing of arbitrary positional arguments rather than just some number
    of trailing positional arguments.

    Args:
        func: Function from which partial function is to be derived.
        indices: Tuple of indices of positional args of `func` that are
           to be fixed to the values specified in `fixargs`.
        *fixargs: Fixed values for specified positional arguments.
        **fixkwargs: Fixed values for keyword arguments.

    Returns:
       The partial function with fixed arguments.
    """

    def pfunc(*freeargs, **freekwargs):
        numargs = len(fixargs) + len(freeargs)
        args = [
            None,
        ] * numargs
        kfix = 0
        kfree = 0
        for k in range(numargs):
            if k in indices:
                args[k] = fixargs[kfix]
                kfix += 1
            else:
                args[k] = freeargs[kfree]
                kfree += 1
        kwargs = freekwargs.copy()
        kwargs.update(fixkwargs)
        return func(*args, **kwargs)

    posdoc = ""
    if indices:
        posdoc = f"positional arguments {','.join(map(str, indices))}"
    kwdoc = ""
    if fixkwargs:
        kwdoc = f"keyword arguments {','.join(fixkwargs.keys())}"
    pfunc.__doc__ = f"Partial function derived from function {func.__name__}"
    if posdoc or kwdoc:
        pfunc.__doc__ += " by fixing " + (" and ".join(filter(None, (posdoc, kwdoc))))
    return pfunc


def device_info(devid: int = 0) -> str:  # pragma: no cover
    """Get a string describing the specified device.

    Args:
        devid: ID number of device.

    Returns:
        Device description string.
    """
    numdev = jax.device_count()
    if devid >= numdev:
        raise RuntimeError(f"Requested information for device {devid} but only {numdev} present.")
    dev = jax.devices()[devid]
    if dev.platform == "cpu":
        info = "CPU"
    else:
        info = f"{dev.platform.upper()} ({dev.device_kind})"
    return info


def check_for_tracer(func: Callable) -> Callable:
    """Check if positional arguments to `func` are jax tracers.

    This is intended to be used as a decorator for functions that call
    external code from within SCICO. At present, external functions
    cannot be jit-ed or vmap/pmaped. This decorator checks for signs of
    jit/vmap/pmap and raises an appropriate exception.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if any([isinstance(x, DynamicJaxprTracer) for x in args]):
            raise TypeError(
                f"DynamicJaxprTracer found in {func.__name__};  did you jit this function?"
            )
        if any([isinstance(x, BatchTracer) for x in args]):
            raise TypeError(
                f"BatchTracer found in {func.__name__};  did you vmap/pmap this function?"
            )
        return func(*args, **kwargs)

    return wrapper


def url_get(url: str, maxtry: int = 3, timeout: int = 10) -> io.BytesIO:  # pragma: no cover
    """Get content of a file via a URL.

    Args:
        url: URL of the file to be downloaded.
        maxtry: Maximum number of download retries.
        timeout: Timeout in seconds for blocking operations.

    Returns:
        Buffered I/O stream.

    Raises:
        ValueError: If the maxtry parameter is not greater than zero.
        urllib.error.URLError: If the file cannot be downloaded.
    """

    if maxtry <= 0:
        raise ValueError("Parameter maxtry should be greater than zero.")
    for ntry in range(maxtry):
        try:
            rspns = urlrequest.urlopen(url, timeout=timeout)
            cntnt = rspns.read()
            break
        except urlerror.URLError as e:
            if not isinstance(e.reason, socket.timeout):
                raise

    return io.BytesIO(cntnt)



def create_roi_indices(Nx, Ny, roi_start_row, roi_end_row, roi_start_col, roi_end_col, display=False):
    '''
    Create ROI indices for a row and column range used in the scico MBIRJAX wrapper.
    
    Args:
        Nx: Image width
        Ny: Image height
        roi_start_row: Start row index of the ROI
        roi_end_row: End row index of the ROI
        roi_start_col: Start column index of the ROI
        roi_end_col: End column index of the ROI
        display: Whether to display the ROI indices, default is False
    
    Returns:
        roi_indices: ROI indices
    '''
    # Check the validity of the ROI indices
    if roi_start_row < 0 or roi_end_row > Nx or roi_start_col < 0 or roi_end_col > Ny:
        raise ValueError("ROI start and end indices must be within the image dimensions")
    if roi_start_row >= roi_end_row or roi_start_col >= roi_end_col:
        raise ValueError("ROI start indices must be strictly less than end indices")

    # Create ROI indices
    roi_rows = np.arange(roi_start_row, roi_end_row)
    roi_cols = np.arange(roi_start_col, roi_end_col)
    roi_row_grid, roi_col_grid = np.meshgrid(roi_rows, roi_cols, indexing='ij')
    roi_indices = np.ravel_multi_index((roi_row_grid.flatten(), roi_col_grid.flatten()), (Nx, Ny))
    
    if display:
        print(f"ROI indices: {roi_indices}")
        print(f"ROI indices shape: {roi_indices.shape}")
        print(f"ROI coverage: {len(roi_indices)} / {Nx * Ny} = {len(roi_indices) / (Nx * Ny) * 100:.1f}%")
    
    return roi_indices


# Timer classes are copied from https://github.com/bwohlberg/sporco


class Timer:
    """Timer class supporting multiple independent labeled timers.

    The timer is based on the relative time returned by
    :func:`timeit.default_timer`.
    """

    def __init__(
        self,
        labels: Optional[Union[str, List[str]]] = None,
        default_label: str = "main",
        all_label: str = "all",
    ):
        """
        Args:
            labels: Label(s) of the timer(s) to be initialised to zero.
            default_label: Default timer label to be used when methods
                are called without specifying a label.
            all_label: Label string that will be used to denote all
                timer labels.
        """

        # Initialise current and accumulated time dictionaries
        self.t0: Dict[str, Optional[float]] = {}
        self.td: Dict[str, float] = {}
        # Record default label and string indicating all labels
        self.default_label = default_label
        self.all_label = all_label
        # Initialise dictionary entries for labels to be created
        # immediately
        if labels is not None:
            if not isinstance(labels, (list, tuple)):
                labels = [
                    labels,
                ]
            for lbl in labels:
                self.td[lbl] = 0.0
                self.t0[lbl] = None

    def start(self, labels: Optional[Union[str, List[str]]] = None):
        """Start specified timer(s).

        Args:
            labels: Label(s) of the timer(s) to be started. If it is
               ``None``, start the default timer with label specified by
               the `default_label` parameter of :meth:`__init__`.
        """

        # Default label is self.default_label
        if labels is None:
            labels = self.default_label
        # If label is not a list or tuple, create a singleton list
        # containing it
        if not isinstance(labels, (list, tuple)):
            labels = [
                labels,
            ]
        # Iterate over specified label(s)
        t = timer()
        for lbl in labels:
            # On first call to start for a label, set its accumulator to zero
            if lbl not in self.td:
                self.td[lbl] = 0.0
                self.t0[lbl] = None
            # Record the time at which start was called for this lbl if
            # it isn't already running
            if self.t0[lbl] is None:
                self.t0[lbl] = t

    def stop(self, labels: Optional[Union[str, List[str]]] = None):
        """Stop specified timer(s).

        Args:
            labels: Label(s) of the timer(s) to be stopped. If it is
               ``None``, stop the default timer with label specified by
               the `default_label` parameter of :meth:`__init__`. If it
               is equal to the string specified by the `all_label`
               parameter of :meth:`__init__`, stop all timers.
        """

        # Get current time
        t = timer()
        # Default label is self.default_label
        if labels is None:
            labels = self.default_label
        # All timers are affected if label is equal to self.all_label,
        # otherwise only the timer(s) specified by label
        if labels == self.all_label:
            labels = list(self.t0.keys())
        elif not isinstance(labels, (list, tuple)):
            labels = [
                labels,
            ]
        # Iterate over specified label(s)
        for lbl in labels:
            if lbl not in self.t0:
                raise KeyError(f"Unrecognized timer key {lbl}.")
            # If self.t0[lbl] is None, the corresponding timer is
            # already stopped, so no action is required
            if self.t0[lbl] is not None:
                # Increment time accumulator from the elapsed time
                # since most recent start call
                self.td[lbl] += t - self.t0[lbl]  # type: ignore
                # Set start time to None to indicate timer is not running
                self.t0[lbl] = None

    def reset(self, labels: Optional[Union[str, List[str]]] = None):
        """Reset specified timer(s).

        Args:
            labels: Label(s) of the timer(s) to be stopped. If it is
                ``None``, stop the default timer with label specified by
                the `default_label` parameter of :meth:`__init__`. If it
                is equal to the string specified by the `all_label`
                parameter of :meth:`__init__`, stop all timers.
        """

        # Default label is self.default_label
        if labels is None:
            labels = self.default_label
        # All timers are affected if label is equal to self.all_label,
        # otherwise only the timer(s) specified by label
        if labels == self.all_label:
            labels = list(self.t0.keys())
        elif not isinstance(labels, (list, tuple)):
            labels = [
                labels,
            ]
        # Iterate over specified label(s)
        for lbl in labels:
            if lbl not in self.t0:
                raise KeyError(f"Unrecognized timer key {lbl}.")
            # Set start time to None to indicate timer is not running
            self.t0[lbl] = None
            # Set time accumulator to zero
            self.td[lbl] = 0.0

    def elapsed(self, label: Optional[str] = None, total: bool = True) -> float:
        """Get elapsed time since timer start.

        Args:
           label: Label of the timer for which the elapsed time is
               required. If it is ``None``, the default timer with label
               specified by the `default_label` parameter of
               :meth:`__init__` is selected.
           total: If ``True`` return the total elapsed time since the
               first call of :meth:`start` for the selected timer,
               otherwise return the elapsed time since the most recent
               call of :meth:`start` for which there has not been a
               corresponding call to :meth:`stop`.

        Returns:
           Elapsed time.
        """

        # Get current time
        t = timer()
        # Default label is self.default_label
        if label is None:
            label = self.default_label
            # Return 0.0 if default timer selected and it is not initialised
            if label not in self.t0:
                return 0.0
        # Raise exception if timer with specified label does not exist
        if label not in self.t0:
            raise KeyError(f"Unrecognized timer key {label}.")
        # If total flag is True return sum of accumulated time from
        # previous start/stop calls and current start call, otherwise
        # return just the time since the current start call
        te = 0.0
        if self.t0[label] is not None:
            te = t - self.t0[label]  # type: ignore
        if total:
            te += self.td[label]

        return te

    def labels(self) -> List[str]:
        """Get a list of timer labels.

        Returns:
          List of timer labels.
        """

        return list(self.t0.keys())

    def __str__(self) -> str:
        """Return string representation of object.

        The representation consists of a table with the following columns:

          * Timer label.
          * Accumulated time from past start/stop calls.
          * Time since current start call, or 'Stopped' if timer is not
            currently running.
        """

        # Get current time
        t = timer()
        # Length of label field, calculated from max label length
        fldlen = [len(lbl) for lbl in self.t0] + [
            len(self.default_label),
        ]
        lfldln = max(fldlen) + 2
        # Header string for table of timers
        s = f"{'Label':{lfldln}s}  Accum.       Current\n"
        s += "-" * (lfldln + 25) + "\n"
        # Construct table of timer details
        for lbl in sorted(self.t0):
            td = self.td[lbl]
            if self.t0[lbl] is None:
                ts = " Stopped"
            else:
                ts = f" {(t - self.t0[lbl]):.2e} s" % (t - self.t0[lbl])  # type: ignore
            s += f"{lbl:{lfldln}s}  {td:.2e} s  {ts}\n"

        return s


class ContextTimer:
    """A wrapper class for :class:`Timer` that enables its use as a
    context manager.

    For example, instead of

    >>> t = Timer()
    >>> t.start()
    >>> x = sum(range(1000))
    >>> t.stop()
    >>> elapsed = t.elapsed()

    one can use

    >>> t = Timer()
    >>> with ContextTimer(t):
    ...   x = sum(range(1000))
    >>> elapsed = t.elapsed()
    """

    def __init__(
        self,
        timer: Optional[Timer] = None,
        label: Optional[str] = None,
        action: str = "StartStop",
    ):
        """
        Args:
           timer: Timer object to be used as a context manager. If
              ``None``, a new class:`Timer` object is constructed.
           label: Label of the timer to be used. If it is ``None``, start
              the default timer.
           action: Actions to be taken on context entry and exit. If the
              value is 'StartStop', start the timer on entry and stop on
              exit; if it is 'StopStart', stop the timer on entry and
              start it on exit.
        """

        if action not in ["StartStop", "StopStart"]:
            raise ValueError(f"Unrecognized action {action}.")
        if timer is None:
            self.timer = Timer()
        else:
            self.timer = timer
        self.label = label
        self.action = action

    def __enter__(self):
        """Start the timer and return this ContextTimer instance."""

        if self.action == "StartStop":
            self.timer.start(self.label)
        else:
            self.timer.stop(self.label)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Stop the timer and return ``True`` if no exception was raised
        within the `with` block, otherwise return ``False``.
        """

        if self.action == "StartStop":
            self.timer.stop(self.label)
        else:
            self.timer.start(self.label)
        return not exc_type

    def elapsed(self, total: bool = True) -> float:
        """Return the elapsed time for the timer.

        Args:
          total: If ``True`` return the total elapsed time since the
             first call of :meth:`start` for the selected timer,
             otherwise return the elapsed time since the most recent call
             of :meth:`start` for which there has not been a
             corresponding call to :meth:`stop`.

        Returns:
          Elapsed time.
        """

        return self.timer.elapsed(self.label, total=total)
