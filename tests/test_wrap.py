import mpl_wrap


def test_version():
    assert mpl_wrap.__version__ == "0.1.0"


def test_public_api():
    for name in (
        "set_wrap",
        "plot_wrapped",
        "scatter_wrapped",
        "fill_between_wrapped",
        "stairs_wrapped",
        "errorbar_wrapped",
    ):
        assert callable(getattr(mpl_wrap, name))
