# Copyright (C) 2020-2024 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Linear operator base class."""


# Needed to annotate a class method that returns the encapsulating class;
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

from functools import wraps
from typing import Callable, Optional, Union

import numpy as np

import jax
import jax.numpy as jnp
from jax.dtypes import result_type

import scico.numpy as snp
from scico._autograd import linear_adjoint
from scico.numpy import Array, BlockArray
from scico.numpy.util import is_complex_dtype
from scico.operator._operator import Operator, _wrap_mul_div_scalar
from scico.typing import BlockShape, DType, Shape

import torch


def _wrap_add_sub(func: Callable) -> Callable:
    r"""Wrapper function for defining `__add__` and `__sub__`.

    Wrapper function for defining `__add__` and ` __sub__` between
    :class:`LinearOperator` and derived classes. Operations
    between :class:`LinearOperator` and :class:`.Operator`
    types are also supported.

    Handles shape checking and function dispatch based on types of
    operands `a` and `b` in the call `func(a, b)`. Note that `func`
    will always be a method of the type of `a`, and since this wrapper
    should only be applied within :class:`LinearOperator` or derived
    classes, we can assume that `a` is always an instance of
    :class:`LinearOperator`. The general rule for dispatch is that the
    `__add__` or `__sub__` operator of the nearest common base class
    of `a` and `b` should be called. If `b` is derived from `a`, this
    entails using the operator defined in the class of `a`, and
    vice-versa. If one of the operands is not a descendant of the other
    in the class hierarchy, then it is assumed that their common base
    class is either :class:`.Operator` or :class:`LinearOperator`,
    depending on the type of `b`.

    - If `b` is not an instance of :class:`.Operator`, a :exc:`TypeError`
      is raised.
    - If the shapes of `a` and `b` do not match, a :exc:`ValueError` is
      raised.
    - If `b` is an instance of the type of `a` then `func(a, b)` is
      called where `func` is the argument of this wrapper, i.e.
      the unwrapped function defined in the class of `a`.
    - If `a` is an instance of the type of `b` then `func(a, b)` is
      called where `func` is the unwrapped function defined in the class
      of `b`.
    - If `b` is a :class:`LinearOperator` then `func(a, b)` is called
      where `func` is the operator defined in :class:`LinearOperator`.
    - Othwerwise, `func(a, b)` is called where `func` is the operator
      defined in :class:`.Operator`.

    Args:
        func: should be either `.__add__` or `.__sub__`.

    Returns:
       Wrapped version of `func`.

    Raises:
        ValueError: If the shapes of two operators do not match.
        TypeError: If one of the two operands is not an
            :class:`.Operator` or :class:`LinearOperator`.
    """

    @wraps(func)
    def wrapper(
        a: LinearOperator, b: Union[Operator, LinearOperator]
    ) -> Union[Operator, LinearOperator]:
        if isinstance(b, Operator):
            if a.shape == b.shape:
                if isinstance(b, type(a)):
                    # b is an instance of the class of a: call the unwrapped operator
                    # defined in the class of a, which is the func argument of this
                    # wrapper
                    return func(a, b)
                if isinstance(a, type(b)):
                    # a is an instance of class b: call the unwrapped operator
                    # defined in the class of b. A test is required because
                    # the operators defined in Operator and non-LinearOperator
                    # derived classes are not wrapped.
                    if hasattr(getattr(type(b), func.__name__), "_unwrapped"):
                        uwfunc = getattr(type(b), func.__name__)._unwrapped
                    else:
                        uwfunc = getattr(type(b), func.__name__)
                    return uwfunc(a, b)
                # The most general approach here would be to automatically determine
                # the nearest common ancestor of the classes of a and b (e.g. as
                # discussed in https://stackoverflow.com/a/58290475 ), but the
                # simpler approach adopted here is to just assume that the common
                # base of two classes that do not have an ancestor-descendant
                # relationship is either Operator or LinearOperator.
                if isinstance(b, LinearOperator):
                    # LinearOperator + LinearOperator -> LinearOperator
                    uwfunc = getattr(LinearOperator, func.__name__)._unwrapped
                    return uwfunc(a, b)
                # LinearOperator + Operator -> Operator (access to the function
                # definition differs from that for LinearOperator because
                # Operator __add__ and __sub__ are not wrapped)
                uwfunc = getattr(Operator, func.__name__)
                return uwfunc(a, b)
            raise ValueError(f"Shapes {a.shape} and {b.shape} do not match.")
        raise TypeError(f"Operation {func.__name__} not defined between {type(a)} and {type(b)}.")

    wrapper._unwrapped = func  # type: ignore

    return wrapper


class LinearOperator(Operator):
    """Generic linear operator base class"""

    def __init__(
        self,
        input_shape: Union[Shape, BlockShape],
        output_shape: Optional[Union[Shape, BlockShape]] = None,
        eval_fn: Optional[Callable] = None,
        adj_fn: Optional[Callable] = None,
        input_dtype: DType = np.float32,
        output_dtype: Optional[DType] = None,
        jit: bool = False,
    ):
        r"""
        Args:
            input_shape: Shape of input array.
            output_shape: Shape of output array. Defaults to ``None``.
                If ``None``, `output_shape` is determined by evaluating
                `self.__call__` on an input array of zeros.
            eval_fn: Function used in evaluating this
                :class:`LinearOperator`. Defaults to ``None``. If
                ``None``, then `self.__call__` must be defined in any
                derived classes.
            adj_fn: Function used to evaluate the adjoint of this
                :class:`LinearOperator`. Defaults to ``None``. If
                ``None``, the adjoint is not set, and the
                :meth:`._set_adjoint` will be called silently at the
                first :meth:`.adj` call or can be called manually.
            input_dtype: `dtype` for input argument. Defaults to
                :attr:`~numpy.float32`. If the :class:`.LinearOperator`
                implements complex-valued operations, this must be a
                complex dtype (typically :attr:`~numpy.complex64`) for
                correct adjoint and gradient calculation.
            output_dtype: `dtype` for output argument. Defaults to
                ``None``. If ``None``, `output_dtype` is determined by
                evaluating `self.__call__` on an input array of zeros.
            jit: If ``True``, call :meth:`.jit()` on this
                :class:`LinearOperator` to jit the forward, adjoint, and
                gram functions. Same as calling :meth:`.jit` after the
                :class:`LinearOperator` is created.
        """

        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            eval_fn=eval_fn,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            jit=False,
        )

        if not hasattr(self, "_adj"):
            self._adj: Optional[Callable] = None
        if not hasattr(self, "_gram"):
            self._gram: Optional[Callable] = None
        if callable(adj_fn):
            self._adj = adj_fn
            self._gram = lambda x: self.adj(self(x))
        elif adj_fn is not None:
            raise TypeError(f"Parameter adj_fn must be either a Callable or None; got {adj_fn}.")

        if jit:
            self.jit()

    def _set_adjoint(self):
        """Automatically create adjoint method."""
        adj_fun = linear_adjoint(self.__call__, snp.zeros(self.input_shape, dtype=self.input_dtype))
        self._adj = lambda x: adj_fun(x)[0]

    def _set_gram(self):
        """Automatically create gram method."""
        self._gram = lambda x: self.adj(self(x))

    def jit(self):
        """Replace the private functions :meth:`._eval`, :meth:`_adj`, :meth:`._gram`
        with jitted versions.
        """
        if self._adj is None:
            self._set_adjoint()

        if self._gram is None:
            self._set_gram()

        self._eval = jax.jit(self._eval)
        self._adj = jax.jit(self._adj)
        self._gram = jax.jit(self._gram)

    @_wrap_add_sub
    def __add__(self, other):
        return LinearOperator(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            eval_fn=lambda x: self(x) + other(x),
            adj_fn=lambda x: self.adj(x) + other.adj(x),
            input_dtype=self.input_dtype,
            output_dtype=result_type(self.output_dtype, other.output_dtype),
        )

    @_wrap_add_sub
    def __sub__(self, other):
        return LinearOperator(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            eval_fn=lambda x: self(x) - other(x),
            adj_fn=lambda x: self.adj(x) - other.adj(x),
            input_dtype=self.input_dtype,
            output_dtype=result_type(self.output_dtype, other.output_dtype),
        )

    @_wrap_mul_div_scalar
    def __mul__(self, other):
        return LinearOperator(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            eval_fn=lambda x: other * self(x),
            adj_fn=lambda x: snp.conj(other) * self.adj(x),
            input_dtype=self.input_dtype,
            output_dtype=result_type(self.output_dtype, other),
        )

    @_wrap_mul_div_scalar
    def __rmul__(self, other):
        return self.__mul__(other)  # scalar multiplication is commutative

    @_wrap_mul_div_scalar
    def __truediv__(self, other):
        return LinearOperator(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            eval_fn=lambda x: self(x) / other,
            adj_fn=lambda x: self.adj(x) / snp.conj(other),
            input_dtype=self.input_dtype,
            output_dtype=result_type(self.output_dtype, other),
        )

    def __matmul__(self, other):
        # self @ other
        return self(other)

    def __rmatmul__(self, other):
        # other @ self
        if isinstance(other, LinearOperator):
            return other(self)

        if isinstance(other, (np.ndarray, jnp.ndarray)):
            # for real valued inputs: y @ self == (self.T @ y.T).T
            # for complex:  y @ self == (self.conj().T @ y.conj().T).conj().T
            # self.conj().T == self.adj
            return self.adj(other.conj().T).conj().T

        raise NotImplementedError(
            f"Operation __rmatmul__ not defined between {type(self)} and {type(other)}."
        )

    def __call__(
        self, x: Union[LinearOperator, Array, BlockArray]
    ) -> Union[LinearOperator, Array, BlockArray]:
        r"""Evaluate this :class:`LinearOperator` at the point :math:`\mb{x}`.

        Args:
            x: Point at which to evaluate this :class:`LinearOperator`.
               If `x` is a :class:`jax.Array` or :class:`.BlockArray`,
               must have `shape == self.input_shape`. If `x` is a
               :class:`LinearOperator`, must have
               `x.output_shape == self.input_shape`.
        """

        if isinstance(x, LinearOperator):
            return ComposedLinearOperator(self, x)
        # Use Operator __call__ for LinearOperator @ array or LinearOperator @ Operator
        return super().__call__(x)

    def adj(
        self, y: Union[LinearOperator, Array, BlockArray]
    ) -> Union[LinearOperator, Array, BlockArray]:
        """Adjoint of this :class:`LinearOperator`.

        Compute the adjoint of this :class:`LinearOperator` applied to
        input `y`.

        Args:
            y: Point at which to compute adjoint. If `y` is
                :class:`jax.Array` or :class:`.BlockArray`, must have
                `shape == self.output_shape`. If `y` is a
                :class:`LinearOperator`, must have
                `y.output_shape == self.output_shape`.

        Returns:
            Adjoint evaluated at `y`.
        """
        if self._adj is None:
            self._set_adjoint()

        # if isinstance(y, LinearOperator):
        #     return ComposedLinearOperator(self.H, y)
        # if self.output_dtype != y.dtype:
        #     raise ValueError(f"Dtype error: expected {self.output_dtype}, got {y.dtype}.")
        # if self.output_shape != y.shape:
        #     raise ValueError(
        #         f"""Shapes do not conform: input array with shape {y.shape} does not match
        #         LinearOperator output_shape {self.output_shape}."""
        #     )
        # assert self._adj is not None
        
        return self._adj(y)

    @property
    def T(self) -> LinearOperator:
        """Transpose of this :class:`LinearOperator`.

        Return a new :class:`LinearOperator` that implements the
        transpose of this :class:`LinearOperator`. For a real-valued
        :class:`LinearOperator` `A` (`A.input_dtype` is
        :attr:`~numpy.float32` or :attr:`~numpy.float64`), the
        :class:`LinearOperator` `A.T` implements the adjoint:
        `A.T(y) == A.adj(y)`. For a complex-valued :class:`LinearOperator`
        `A` (`A.input_dtype` is :attr:`~numpy.complex64` or
        :attr:`~numpy.complex128`), the :class:`LinearOperator` `A.T` is
        not the adjoint. For the conjugate transpose, use `.conj().T` or
        :meth:`.H`.
        """
        if is_complex_dtype(self.input_dtype):
            return LinearOperator(
                input_shape=self.output_shape,
                output_shape=self.input_shape,
                eval_fn=lambda x: self.adj(x.conj()).conj(),
                adj_fn=self.__call__,
                input_dtype=self.input_dtype,
                output_dtype=self.output_dtype,
            )
        return LinearOperator(
            input_shape=self.output_shape,
            output_shape=self.input_shape,
            eval_fn=self.adj,
            adj_fn=self.__call__,
            input_dtype=self.output_dtype,
            output_dtype=self.input_dtype,
        )

    @property
    def H(self) -> LinearOperator:
        """Hermitian transpose of this :class:`LinearOperator`.

        Return a new :class:`LinearOperator` that is the Hermitian
        transpose of this :class:`LinearOperator`. For a real-valued
        :class:`LinearOperator` `A` (`A.input_dtype` is
        :attr:`~numpy.float32` or :attr:`~numpy.float64`), the
        :class:`LinearOperator` `A.H` is equivalent to `A.T`. For a
        complex-valued :class:`LinearOperator` `A` (`A.input_dtype` is
        :attr:`~numpy.complex64` or :attr:`~numpy.complex128`), the
        :class:`LinearOperator` `A.H` implements the adjoint of `A :
        A.H @ y == A.adj(y) == A.conj().T @ y)`.

        For the non-conjugate transpose, see :meth:`.T`.
        """
        return LinearOperator(
            input_shape=self.output_shape,
            output_shape=self.input_shape,
            eval_fn=self.adj,
            adj_fn=self.__call__,
            input_dtype=self.output_dtype,
            output_dtype=self.input_dtype,
        )

    def conj(self) -> LinearOperator:
        """Complex conjugate of this :class:`LinearOperator`.

        Return a new :class:`LinearOperator` `Ac` such that
        `Ac(x) = conj(A)(x)`.
        """
        # A.conj() x == (A @ x.conj()).conj()
        return LinearOperator(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            eval_fn=lambda x: self(x.conj()).conj(),
            adj_fn=lambda x: self.adj(x.conj()).conj(),
            input_dtype=self.input_dtype,
            output_dtype=self.output_dtype,
        )

    @property
    def gram_op(self) -> LinearOperator:
        """Gram operator of this :class:`LinearOperator`.

        Return a new :class:`LinearOperator` `G` such that
        `G(x) = A.adj(A(x)))`.
        """
        if self._gram is None:
            self._set_gram()

        return LinearOperator(
            input_shape=self.input_shape,
            output_shape=self.input_shape,
            eval_fn=self.gram,
            adj_fn=self.gram,
            input_dtype=self.input_dtype,
            output_dtype=self.output_dtype,
        )

    def gram(
        self, x: Union[LinearOperator, Array, BlockArray]
    ) -> Union[LinearOperator, Array, BlockArray]:
        """Compute `A.adj(A(x)).`

        Args:
            x: Point at which to evaluate the gram operator. If `x` is
               a :class:`jax.Array` or :class:`.BlockArray`, must have
               `shape == self.input_shape`. If `x` is a
               :class:`LinearOperator`, must have
               `x.output_shape == self.input_shape`.

        Returns:
            Result of `A.adj(A(x))`.
        """
        if self._gram is None:
            self._set_gram()
        assert self._gram is not None
        return self._gram(x)


class ComposedLinearOperator(LinearOperator):
    """A composition of two :class:`LinearOperator` objects.

    A new :class:`LinearOperator` formed by the composition of two other
    :class:`LinearOperator` objects.
    """

    def __init__(self, A: LinearOperator, B: LinearOperator, jit: bool = False):
        r"""
        A :class:`ComposedLinearOperator` `AB` implements
        `AB @ x == A @ B @ x`. :class:`LinearOperator` `A` and `B` are
        stored as attributes of the :class:`ComposedLinearOperator`.

        :class:`LinearOperator` `A` and `B` must have compatible shapes
        and dtypes: `A.input_shape == B.output_shape` and
        `A.input_dtype == B.input_dtype`.

        Args:
            A: First (left) :class:`LinearOperator`.
            B: Second (right) :class:`LinearOperator`.
            jit: If ``True``, call :meth:`~.LinearOperator.jit()` on this
                :class:`LinearOperator` to jit the forward, adjoint, and
                gram functions. Same as calling
                :meth:`~.LinearOperator.jit` after the
                :class:`LinearOperator` is created.
        """
        if not isinstance(A, LinearOperator):
            raise TypeError(
                "The first argument to ComposedLinearOperator must be a LinearOperator; "
                f"got {type(A)}."
            )
        if not isinstance(B, LinearOperator):
            raise TypeError(
                "The second argument to ComposedLinearOperator must be a LinearOperator; "
                f"got {type(B)}."
            )
        if A.input_shape != B.output_shape:
            raise ValueError(f"Incompatable LinearOperator shapes {A.shape}, {B.shape}.")
        if A.input_dtype != B.output_dtype:
            raise ValueError(
                f"Incompatable LinearOperator dtypes {A.input_dtype}, {B.output_dtype}."
            )

        self.A = A
        self.B = B

        super().__init__(
            input_shape=self.B.input_shape,
            output_shape=self.A.output_shape,
            input_dtype=self.B.input_dtype,
            output_dtype=self.A.output_dtype,
            eval_fn=lambda x: self.A(self.B(x)),
            adj_fn=lambda z: self.B.adj(self.A.adj(z)),
            jit=jit,
        )
