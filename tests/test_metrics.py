import numpy as np

from src.evaluation.metrics import cer, der, sdri, si_snr, si_snri, wer
from src.types import SpeakerTurn


def test_der_identical_is_zero():
    ref = [SpeakerTurn(0, 2, "A"), SpeakerTurn(2, 4, "B")]
    assert der(ref, ref)["der"] == 0.0


def test_der_invariant_to_label_names():
    ref = [SpeakerTurn(0, 2, "A"), SpeakerTurn(2, 4, "B")]
    hyp = [SpeakerTurn(0, 2, "SPEAKER_7"), SpeakerTurn(2, 4, "SPEAKER_3")]
    assert der(ref, hyp)["der"] == 0.0


def test_der_penalizes_confusion():
    ref = [SpeakerTurn(0, 2, "A"), SpeakerTurn(2, 4, "B")]
    hyp = [SpeakerTurn(0, 4, "A")]  # merged both speakers into one
    assert der(ref, hyp)["der"] > 0.0


def test_si_snr_perfect_is_high():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(16000).astype(np.float32)
    assert si_snr(ref, ref) > 50


def test_si_snri_positive_when_estimate_better_than_mixture():
    rng = np.random.default_rng(1)
    ref = rng.standard_normal(16000).astype(np.float32)
    interferer = rng.standard_normal(16000).astype(np.float32)
    mixture = ref + interferer
    estimate = ref + 0.05 * interferer  # much closer to ref than the mixture
    assert si_snri(estimate, ref, mixture) > 0
    assert sdri(estimate, ref, mixture) > 0


def test_sdri_is_scale_invariant():
    # neural separators output arbitrary scale; SDRi must not change when the estimate is rescaled
    rng = np.random.default_rng(3)
    ref = rng.standard_normal(16000).astype(np.float32)
    interferer = rng.standard_normal(16000).astype(np.float32)
    mixture = ref + interferer
    estimate = ref + 0.1 * interferer
    base = sdri(estimate, ref, mixture)
    assert abs(sdri(10.0 * estimate, ref, mixture) - base) < 1e-3
    assert base > 0


def test_wer_cer_basic():
    assert wer("hello world", "hello world") == 0.0
    assert wer("a b c d", "a b x d") == 0.25
    assert cer("hello world", "hello world") == 0.0


def test_wer_devanagari_nfc():
    # same string in two Unicode normal forms must score 0 under NFC normalization
    import unicodedata
    s = "नमस्ते दुनिया"
    assert wer(unicodedata.normalize("NFD", s), unicodedata.normalize("NFC", s)) == 0.0
