from enum import Enum
from typing import Any, Dict, Optional, Protocol, cast

import numpy as np
from scipy.stats import exponweib, gamma

from utils.types import NPFArray, f32


class DistributionType(Enum):
    EXPWEIB = 1
    GAMMA = 2


class PdfFitter(Protocol):
    """Structural type for step-density fitters used by the trajectory
    analyzers: anything with a matching `pdf(x)` works, real fitters and
    test doubles alike."""

    def pdf(self, x: NPFArray) -> NPFArray: ...


class WeibullFitter:
    ftype: DistributionType = DistributionType.EXPWEIB

    def __init__(self) -> None:
        self.params: Optional[tuple[float, float, float, float]] = None

    def fit(self, data: NPFArray) -> None:
        assert self.params is None
        # exponweib.fit возвращает (a, c, loc, scale)
        fitted = [float(v) for v in exponweib.fit(data)]
        self.params = (fitted[0], fitted[1], fitted[2], fitted[3])

    def rvs(self, size: int) -> NPFArray:
        assert self.params is not None
        a, c, loc, scale = self.params
        # это использует scipy/numpy RNG внутри, векторно
        return cast(
            NPFArray,
            exponweib.rvs(a, c, loc=loc, scale=scale, size=size).astype(f32),
        )

    def pdf(self, x: NPFArray) -> NPFArray:
        assert self.params is not None
        a, c, loc, scale = self.params
        return cast(
            NPFArray, exponweib.pdf(x, a, c, loc=loc, scale=scale).astype(f32)
        )

    def cdf(self, x: NPFArray) -> NPFArray:
        assert self.params is not None
        a, c, loc, scale = self.params
        return cast(
            NPFArray, exponweib.cdf(x, a, c, loc=loc, scale=scale).astype(f32)
        )

    def to_dict(self) -> Dict[str, Any]:
        assert self.params is not None
        return {"ftype": self.ftype.name, "params": list(self.params)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WeibullFitter":
        fitter = cls()
        a, c, loc, scale = data["params"]
        fitter.params = (float(a), float(c), float(loc), float(scale))
        return fitter


class GammaFitter:
    ftype: DistributionType = DistributionType.GAMMA

    def __init__(self) -> None:
        self.params: Optional[NPFArray] = None

    def fit(self, data: NPFArray) -> None:
        assert self.params is None
        self.params = gamma.fit(data)

    @staticmethod
    def interp(val: float, x: NPFArray, csum: NPFArray) -> float:
        if val > csum[-1]:
            return float(x[-1])
        if val < csum[0]:
            return float(x[0])
        mask = np.where(val > csum)[0]
        i1 = mask[-1]
        i2 = i1 + 1
        return float(
            x[i1] + (x[i2] - x[i1]) * (val - csum[i1]) / (csum[i2] - csum[i1])
        )

    def pdf(self, x: NPFArray) -> NPFArray:
        assert self.params is not None
        return cast(NPFArray, gamma.pdf(x, *self.params))

    def cdf(self, x: NPFArray) -> NPFArray:
        assert self.params is not None
        return cast(NPFArray, gamma.cdf(x, *self.params))

    def to_dict(self) -> Dict[str, Any]:
        assert self.params is not None
        return {
            "ftype": self.ftype.name,
            "params": [float(p) for p in self.params],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GammaFitter":
        fitter = cls()
        fitter.params = np.array(
            [float(p) for p in data["params"]], dtype=np.float64
        )
        return fitter
