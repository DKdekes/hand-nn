from hand.activate import sigmoid


def test_sigmoid():
    print()
    x = 0.5
    expected = 0.6224593312018546
    res = sigmoid.sigmoid(x)
    assert expected == res, 'bad sigmoid calculation'


def test_d_sigmoid():
    x = 0.5
    expected = 0.2350037122015945
    res = sigmoid.d_sigmoid(x)
    assert expected == res, 'bad d_sigmoid calculation'
