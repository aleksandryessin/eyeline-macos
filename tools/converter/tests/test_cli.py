from eyeline_converter.cli import _shape


def test_shape_parser() -> None:
    assert _shape("1,64,64,3") == (1, 64, 64, 3)
