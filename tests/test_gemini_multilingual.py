from types import SimpleNamespace
from unittest.mock import MagicMock

from omegaconf import OmegaConf

from src.asr.sarvam import SarvamASR
from src.postprocessing.gemini import GeminiPostProcessor, _heuristic_detect_language
from src.types import Transcript, TranscriptSegment


def test_heuristic_detect_language():
    # Pure Hindi
    lang, is_cs = _heuristic_detect_language("नमस्ते आप कैसे हैं")
    assert lang == "Hindi"
    assert not is_cs

    # Code-switched Hinglish (Devanagari + English words)
    lang, is_cs = _heuristic_detect_language("आज का meeting बहुत important था")
    assert "Hinglish" in lang
    assert is_cs

    # Bengali
    lang, is_cs = _heuristic_detect_language("আজকের আবহাওয়া খুব সুন্দর")
    assert "Bengali" in lang

    # Pure English
    lang, is_cs = _heuristic_detect_language("Good morning, welcome to our discussion.")
    assert lang == "English"
    assert not is_cs


def test_gemini_postprocessor_offline_fallback():
    cfg = OmegaConf.create({
        "backend": "gemini",
        "model": "gemini-2.0-flash",
        "api_key_env": "NON_EXISTENT_KEY",
    })
    post = GeminiPostProcessor(cfg)
    post.load()
    assert post._client is None

    transcript = Transcript(segments=[
        TranscriptSegment(0.0, 2.5, "SPEAKER_00", "आज का market price क्या है"),
        TranscriptSegment(3.0, 5.0, "SPEAKER_01", "The wheat price is steady today"),
    ])

    notes = []
    report = post.run(transcript, notes)

    assert report.summary
    assert len(report.dialogue) == 2
    assert "Hinglish" in report.dialogue[0].lang or "Hindi" in report.dialogue[0].lang
    assert report.dialogue[1].lang == "English"
    assert any("heuristic" in n.lower() for n in notes)


def test_gemini_postprocessor_with_mocked_gemini_api():
    cfg = OmegaConf.create({
        "backend": "gemini",
        "model": "gemini-2.0-flash",
        "api_key_env": "MOCK_KEY",
    })
    post = GeminiPostProcessor(cfg)

    # Mock client and response
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = """{
      "summary": "Speaker 0 asked about the market price, and Speaker 1 clarified wheat prices in English.",
      "tags": ["Market Prices", "Agriculture", "Wheat"],
      "languages_detected": ["Hinglish (Hindi + English)", "English"],
      "language_breakdown": {"Hindi": "50%", "English": "50%"},
      "segments": [
        {
          "index": 0,
          "speaker": "SPEAKER_00",
          "cleaned_text": "आज का market price क्या है?",
          "lang": "Hinglish (Hindi + English)",
          "is_code_switched": true,
          "translation": "What is today's market price?"
        },
        {
          "index": 1,
          "speaker": "SPEAKER_01",
          "cleaned_text": "The wheat price is steady today.",
          "lang": "English",
          "is_code_switched": false,
          "translation": "The wheat price is steady today."
        }
      ]
    }"""
    mock_client.models.generate_content.return_value = mock_response
    post._client = mock_client

    transcript = Transcript(segments=[
        TranscriptSegment(0.0, 2.5, "SPEAKER_00", "aaj ka market price kya hai"),
        TranscriptSegment(3.0, 5.0, "SPEAKER_01", "The wheat price is steady today"),
    ])

    notes = []
    report = post.run(transcript, notes)

    assert "Speaker 0 asked about the market price" in report.summary
    assert "Market Prices" in report.tags
    assert report.dialogue[0].translation == "What is today's market price?"
    assert report.dialogue[0].is_code_switched is True
    assert "Hinglish" in report.dialogue[0].lang
    assert report.dialogue[1].lang == "English"
    assert report.dialogue[1].is_code_switched is False


def test_sarvam_response_extracts_language_code():
    cfg = OmegaConf.create({"model": "saaras:v4"})
    asr = SarvamASR(cfg)

    fake_response = SimpleNamespace(
        transcript="नमस्ते दुनिया",
        language_code="hi-IN",
        timestamps=SimpleNamespace(
            words=["नमस्ते", "दुनिया"],
            start_time_seconds=[0.1, 0.6],
            end_time_seconds=[0.5, 1.0],
        ),
    )

    segments = asr._response_segments(fake_response, "SPEAKER_00", 1.2, offset=0.0)
    assert len(segments) == 1
    assert segments[0].lang == "Hindi"
    assert segments[0].speaker == "SPEAKER_00"
    assert segments[0].text == "नमस्ते दुनिया"
