from hand.error import squared_error


def test_squared_error():
    y = 1
    y_p = 3
    expect = 4
    res = squared_error.squared_error(y, y_p)
    assert expect == res, 'bad squared error'


def test_d_squared_error():
    y = 2
    y_p = 6
    expect = -8
    res = squared_error.d_squared_error(y, y_p)
    assert expect == res, 'bad derivative of squared error'
