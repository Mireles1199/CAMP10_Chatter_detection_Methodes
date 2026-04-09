def test_api():
    import ssq_chatter as pkg
    assert hasattr(pkg, "detect_chatter_from_force")
    assert hasattr(pkg, "make_chatter_like_signal")
