from scaledp.utils import get_size


def test_get_size():
    # Test with an empty list
    assert get_size([]) == 0

    # Test with a single-element list
    assert get_size([5]) == 5

    # Test with multiple elements
    assert get_size([1, 2, 3, 4, 5]) == 2

    # Test with a key function
    items = [{'value': 1}, {'value': 2}, {'value': 3}, {'value': 4}, {'value': 5}]
    assert get_size(items, key=lambda x: x['value']) == 2

    # Test with a small list
    assert get_size([10, 20, 30, 40]) == 25

    # Test with a large list
    assert get_size([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]) == 4
