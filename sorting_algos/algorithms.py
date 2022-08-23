def bubble_sort(lst: list) -> None:
    changed = True
    while changed:
        changed = False
        for i in range(len(lst) - 1):
            if lst[i] > lst[i + 1]:
                lst[i], lst[i + 1] = lst[i + 1], lst[i]
                changed = True
    return lst


def selection_sort(lst: list) -> None:
    for i, e in enumerate(lst):
        mn = min(range(i, len(lst)), key=lst.__getitem__)
        lst[i], lst[mn] = lst[mn], e
    return lst


def shell_sort(lst: list) -> None:
    inc = len(lst) // 2
    while inc:
        for i, el in enumerate(lst[inc:], inc):
            while i >= inc and lst[i - inc] > el:
                lst[i] = lst[i - inc]
                i -= inc
            lst[i] = el
        inc = 1 if inc == 2 else inc * 5 // 11
    return lst


def quick_sort(lst: list) -> None:
    _quick_sort(lst, 0, len(lst) - 1)
    return lst


def _quick_sort(lst: list, start: int, stop: int) -> None:
    if stop - start > 0:
        pivot, left, right = lst[start], start, stop
        while left <= right:
            while lst[left] < pivot:
                left += 1
            while lst[right] > pivot:
                right -= 1
            if left <= right:
                lst[left], lst[right] = lst[right], lst[left]
                left += 1
                right -= 1
        _quick_sort(lst, start, right)
        _quick_sort(lst, left, stop)
