"""
auto engine
"""
from __future__ import annotations
import typing
__all__ = ['cross_entropy_loss', 'invalid_argument', 'mse_loss', 'op', 'opt_algo', 'runtime_err', 'tensor']
class invalid_argument(Exception):
    pass
class op:
    @staticmethod
    def cat(arg0: list[op], arg1: int) -> op:
        ...
    @typing.overload
    def __add__(self, arg0: op) -> op:
        ...
    @typing.overload
    def __add__(self, arg0: float) -> op:
        ...
    @typing.overload
    def __mul__(self, arg0: op) -> op:
        ...
    @typing.overload
    def __mul__(self, arg0: float) -> op:
        ...
    @typing.overload
    def __pow__(self, arg0: op) -> op:
        ...
    @typing.overload
    def __pow__(self, arg0: float) -> op:
        ...
    def __repr__(self) -> str:
        ...
    @typing.overload
    def __sub__(self, arg0: op) -> op:
        ...
    @typing.overload
    def __sub__(self, arg0: float) -> op:
        ...
    @typing.overload
    def __truediv__(self, arg0: op) -> op:
        ...
    @typing.overload
    def __truediv__(self, arg0: float) -> op:
        ...
    def add_n(self, arg0: op) -> op:
        ...
    def backward(self) -> None:
        ...
    def clear_grad(self) -> None:
        ...
    def clear_grad_graph(self) -> None:
        ...
    def cos(self) -> op:
        ...
    def create_grad_graph(self) -> None:
        ...
    def div_n(self, arg0: op) -> op:
        ...
    @typing.overload
    def expand(self, arg0: list[int]) -> op:
        ...
    @typing.overload
    def expand(self, arg0: list[int], arg1: int) -> op:
        ...
    def grad(self) -> op:
        ...
    def grad_graph(self) -> op:
        ...
    def inverse(self) -> op:
        ...
    def item(self) -> float:
        ...
    def log(self) -> op:
        ...
    def mm(self, arg0: op) -> op:
        ...
    def mul_n(self, arg0: op) -> op:
        ...
    def permute(self, arg0: list[int]) -> op:
        ...
    def pow_n(self, arg0: op) -> op:
        ...
    def reshape(self, arg0: list[int]) -> op:
        ...
    def shape(self) -> list[int]:
        ...
    def sin(self) -> op:
        ...
    def split(self, arg0: list[int], arg1: int) -> list[op]:
        ...
    def sub_n(self, arg0: op) -> op:
        ...
    @typing.overload
    def sum(self) -> op:
        ...
    @typing.overload
    def sum(self, arg0: int) -> op:
        ...
    def tolist(self) -> list:
        ...
    def transpose(self, d1: int = -2, d2: int = -1) -> op:
        ...
    def update(self, arg0: op) -> None:
        ...
class opt_algo:
    def __init__(self, cost_func: typing.Callable[[list[op]], op], vars: list[op]) -> None:
        ...
    def algo_hyper_params(self, algo: str, hyper_params: dict[str, float]) -> None:
        ...
    def run(self) -> None:
        ...
class runtime_err(Exception):
    pass
def cross_entropy_loss(arg0: op, arg1: op, arg2: int) -> op:
    ...
def mse_loss(arg0: op, arg1: op) -> op:
    ...
def tensor(lst: list, requires_grad: bool = False) -> op:
    ...
