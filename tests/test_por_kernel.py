from por_kernel import resonate


def test_resonate_normalizes_signal():
    assert resonate("  Echo ") == "resonance:echo"


def test_resonate_silent_when_empty():
    assert resonate("   ") == "silent"
