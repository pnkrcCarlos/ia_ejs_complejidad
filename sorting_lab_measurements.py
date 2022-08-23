import sys
import operator
import random
import time
from dataclasses import dataclass, field
from itertools import product
from typing import Callable

from sorting_algos.algorithms import bubble_sort, selection_sort, shell_sort, quick_sort
from data_makers.makers import (
    DataMaker,
    UniformIntMaker,
    SortedUniformIntMaker,
    NearlySortedUniformIntMaker,
    AFewIntMaker,
)

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
from pandas import DataFrame, read_hdf


def is_sorted(a: list, *, key=None, reverse=False) -> bool:
    if not a:
        return True
    b = a if key is None else map(key, a)
    cmp = operator.le if not reverse else operator.ge
    head = iter(b)
    tail = iter(b)
    next(tail)
    return all(cmp(x, y) for x, y in zip(head, tail))


@dataclass
class SortingTestCase:
    size: int
    seed: int
    algorithm: Callable
    data_maker: DataMaker
    times_ns: list[int] = field(default_factory=list)

    def as_tuple(self) -> tuple:
        return (
            self.size,
            self.seed,
            self.algorithm.__name__,
            self.data_maker.__class__.__name__,
            self.times_ns,
        )

    def run(self) -> None:
        data = self.data_maker.make(self.size, self.seed)
        print(
            f"Running {self.algorithm.__name__} test with size={self.size}, seed={self.seed}, data_maker={self.data_maker.__class__.__name__} ..."
        )
        start = time.perf_counter_ns()
        self.algorithm(data)
        end = time.perf_counter_ns()
        assert is_sorted(data)

        elapsed = end - start
        self.times_ns.append(elapsed)


def make_all_test_cases(
    sizes: list[int],
    seeds: list[int],
    algorithms: list[Callable],
    data_makers: list[DataMaker],
) -> list[SortingTestCase]:
    tests: list[SortingTestCase] = []

    for size, seed, algorithm, data_maker in product(
        sizes, seeds, algorithms, data_makers
    ):
        tests.append(
            SortingTestCase(
                size=size, seed=seed, algorithm=algorithm, data_maker=data_maker
            )
        )

    return tests


def run_tests_n_times(
    tests: list[SortingTestCase], trials: int, shuffler: random.Random
) -> None:
    for i in range(trials):
        print(f"starting epoch {i + 1}/{trials}")
        # The order in which tests are ran affects results. Tests are shuffled to mitigate that.
        shuffler.shuffle(tests)
        for test in tests:
            test.run()


def test_data_to_df(tests: list[SortingTestCase]) -> DataFrame:
    df = DataFrame.from_records(
        data=map(lambda test: test.as_tuple(), tests),
        columns=["size", "seed", "algorithm", "data_maker", "time"],
    )
    df["algorithm"] = df["algorithm"].astype(str)
    df["data_maker"] = df["data_maker"].astype(str)
    df = df.explode("time")
    df["time"] = df["time"].astype(float)
    df["time"] /= 10**3  # convert to microseconds
    return df


def make_plot(df: DataFrame):
    fig: matplotlib.figure.Figure
    ax: matplotlib.axes.Axes
    fig, ax = plt.subplots()
    ax.set_title("Sorting Times")
    ax.set_xlabel("Number of elements to sort")
    ax.set_ylabel("Time (microseconds)")
    lines = []
    grouped = df.groupby(["algorithm", "data_maker"])
    for name, group in grouped:
        means = group.groupby("size")["time"].mean()
        cls, kwargs = name
        label = str(cls) if kwargs == "None" else f"{cls}, {kwargs}"
        (line,) = ax.plot(means, label=label)
        lines.append(line)

    leg = ax.legend(fancybox=True, shadow=True)
    lined = {}  # maps legend lines to original lines.
    for legline, origline in zip(leg.get_lines(), lines):
        legline.set_picker(True)
        lined[legline] = origline

    def on_pick(event):
        legline = event.artist
        origline = lined[legline]
        visible = not origline.get_visible()
        origline.set_visible(visible)
        legline.set_alpha(1.0 if visible else 0.2)
        ax.relim(visible_only=True)
        ax.autoscale_view()
        fig.canvas.draw()

    fig.canvas.mpl_connect("pick_event", on_pick)
    plt.show()


def main():
    sys.setrecursionlimit(15000)
    compute_results: bool = True
    hdf_filename: str = "sort_times_df.hdf"
    lengths: list[int] = [1024, 4096, 16384]
    seeds: list[int] = list(range(1))
    runs: int = 10
    algorithms: list[Callable] = [bubble_sort, selection_sort, shell_sort, quick_sort]
    data_makers: list[DataMaker] = [
        UniformIntMaker(),
        # SortedUniformIntMaker(),
        # NearlySortedUniformIntMaker(),
        # AFewIntMaker(),
    ]

    if compute_results:
        tests = make_all_test_cases(
            sizes=lengths, seeds=seeds, algorithms=algorithms, data_makers=data_makers
        )
        shuffle_random = random.Random(1)
        run_tests_n_times(tests, runs, shuffle_random)
        df = test_data_to_df(tests)
        df.to_hdf(hdf_filename, "results")
    else:
        df = read_hdf(hdf_filename)
        maker_names = [maker.__class__.__name__ for maker in data_makers]
        df = df[df["data_maker"].isin(maker_names)]

    make_plot(df)


if __name__ == "__main__":
    main()
