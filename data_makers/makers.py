from abc import ABC, abstractmethod
import numpy as np


class DataMaker(ABC):
    @abstractmethod
    def make(self, size, seed) -> list:
        return []

    def compatible_keys(self):
        return [None]

    def compatible_sort_kwargs(self):
        keys = self.compatible_keys()
        kwargs_list = []
        if None in keys:
            kwargs_list += [None, {"reverse": True}]
        kwargs_list += [
            {"key": key, "reverse": True} for key in keys if key is not None
        ]
        kwargs_list += [{"key": key} for key in keys if key is not None]
        return kwargs_list


class IntMaker(DataMaker):
    @abstractmethod
    def make(self, size, seed) -> list[int]:
        return []


class UniformIntMaker(IntMaker):
    def make(self, size, seed) -> list[int]:
        np.random.seed(seed)
        return list(np.random.randint(low=0, high=2**31 - 1, size=size))


class SortedUniformIntMaker(IntMaker):
    def make(self, size, seed):
        np.random.seed(seed)
        return sorted(np.random.randint(low=0, high=2**31 - 1, size=size))


class NearlySortedUniformIntMaker(IntMaker):
    def make(self, size, seed):
        np.random.seed(seed)
        data = sorted(np.random.randint(low=0, high=2**31 - 1, size=size))
        if len(data) < 3:
            return data
        for _ in range(10):
            i, j = np.random.randint(low=0, high=size - 1, size=2)
            data[i], data[j] = data[j], data[i]
        return data


class AFewIntMaker(DataMaker):
    def make(self, size, seed):
        np.random.seed(seed)
        return list(np.random.randint(low=0, high=32, size=size))
