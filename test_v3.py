"""Offline regression tests: python -m unittest discover -s tests -v"""
import importlib.util
import logging
import os
import pathlib
import sys
import types
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).parents[1]
base = types.ModuleType("run_digest")
base.logger = logging.getLogger("test-morning-digest")
base.SESSION = object()
base.LLM_TIMEOUT_SECONDS = 90
base.REQUEST_TIMEOUT_SECONDS = 15
base.MAX_TRANSCRIPT_CHARS = 4500
base.OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
base.canonical_url = lambda s: s
base.html_to_text = lambda s: s
base.fetch_rss_items = lambda *args, **kwargs: []
base.validate_agent_result = lambda result, *args: result
base.build_fallback_digest = lambda **kw: {"provider": "Source-only fallback", "subject": "Fallback", "html_body": "<p>Fallback</p>"}
base.discover_openrouter_free_models = lambda *args: ["qwen/something:free"]
sys.modules["run_digest"] = base
spec = importlib.util.spec_from_file_location("run_digest_v3", ROOT / "run_digest_v3.py")
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

NEWS = [{"id": "news-1", "title": "<Suspicious title>", "url": "https://example.org/one", "published": "2026-11-22", "summary": "Verified story"}]
TOOLS = [{"id": "tool-1", "title": "MCP tool", "url": "https://example.org/two", "published": "2026-11-22", "summary": "Useful tooling"}]
SEC = [{"id": "sec-1", "title": "Agent access", "url": "https://example.org/three", "published": "2026-11-22", "summary": "Security risk"}]
PAPERS = [{"id": "paper-1", "title": "Model paper", "url": "https://example.org/four", "published": "2026-11-22", "summary": "Research paper"}]
VIDEOS = [{"id": "vid-1", "title": "Interview", "url": "https://example.org/five", "published": "2026-11-22", "transcript_available": False}]
ANSWER = {
    "subject": "AI engineering roundup", "takeaway": "Secure tools are important.",
    "news": [{"id": "news-1", "impact": "Engineering impact", "action": "Review"}],
    "interviews": [{"id": "vid-1", "insight": "Title-based only"}],
    "tool": {"id": "tool-1", "problem": "Reduce toil", "implementation": "Explore the docs", "tradeoffs": "Permissions"},
    "security": [{"id": "sec-1", "risk": "Unauthorized access", "mitigation": "Least privilege"}],
    "paper": {"id": "paper-1", "impact": "Potential", "usage": "Evaluate"},
    "build": {"goal": "Review permission limits", "steps": ["Test read only", "Record results"], "done": "Checks pass"},
}


class TestV3(unittest.TestCase):
    def test_nested_json_with_think_and_code_fence(self):
        value = v3.parse_json('<think>{bad}</think>```json\n{"a":{"b":1}}\n```')
        self.assertEqual(value["a"]["b"], 1)

    def test_json_parser_rejects_incomplete(self):
        with self.assertRaises(ValueError):
            v3.parse_json('{"a":')

    def test_text_normalization(self):
        self.assertEqual(v3.extract_text([{"type": "text", "text": "Hi"}, {"text": {"value": "there"}}]), "Hi\nthere")

    def test_compact_prompt_is_bounded(self):
        prompt = v3.compact_prompt(NEWS * 50, TOOLS * 60, SEC * 20, PAPERS * 30, VIDEOS * 10, 4800)
        self.assertLessEqual(len(prompt), 4800)
        self.assertIn("SOURCE DATA JSON", prompt)

    def test_html_escapes_and_preserves_six_sections(self):
        doc = v3.render_digest(ANSWER, NEWS, TOOLS, SEC, PAPERS, VIDEOS)
        self.assertIn("&lt;Suspicious title&gt;", doc["html_body"])
        self.assertNotIn("<Suspicious title>", doc["html_body"])
        self.assertEqual(doc["selected_news_ids"], ["news-1"])
        self.assertEqual(doc["selected_security_ids"], ["sec-1"])
        self.assertEqual(doc["html_body"].count("<section"), 6)

    def test_rejects_invalid_source_only_model_ids(self):
        invented = ANSWER | {"news": [{"id": "faked-link", "impact": "wrong"}], "tool": {},
                             "paper": {}, "interviews": [], "security": []}
        with self.assertRaises(ValueError):
            v3.render_digest(invented, NEWS, TOOLS, SEC, PAPERS, VIDEOS)

    def test_groq_request_has_low_reasoning_json(self):
        with patch.object(v3, "request_json", return_value={"choices": [{"message": {"content": '{"test":1}'}}]}) as req:
            parsed = v3.openai_call(v3.GROQ_CHAT, "fake-key", "openai/gpt-oss-20b", "hello", output_tokens=1500, groq=True)
            self.assertEqual(parsed, {"test": 1})
            payload = req.call_args.kwargs["payload"]
            self.assertEqual(payload["reasoning_format"], "hidden")
            self.assertEqual(payload["response_format"]["type"], "json_object")

    def test_gemini_discovery_skips_unsupported_and_prefer_lite(self):
        listed = {"models": [
            {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-2.5-flash-lite", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-2.0-something", "supportedGenerationMethods": ["countTokens"]},
        ]}
        with patch.object(v3, "request_json", return_value=listed):
            names = v3.gemini_models("fake-key")
            self.assertEqual(names[0], "gemini-2.5-flash-lite")
            self.assertEqual(len(names), 2)

    def test_native_gemini_text(self):
        payload = {"candidates": [{"content": {"parts": [{"text": '{"news": []}', "thought": False}]}}]}
        with patch.object(v3, "request_json", return_value=payload):
            self.assertEqual(v3.gemini_call("key", "gemini-2.5-flash-lite", "prompt", 2000), {"news": []})

    def test_groq_retries_on_413(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "", "GEMINI_API_KEY": "", "GROQ_API_KEY": "fake"}):
            with patch.object(v3, "openai_call", side_effect=[v3.ProviderError(413, "too large"), ANSWER]) as call:
                result = v3.generate({}, PAPERS, NEWS, TOOLS, VIDEOS, SEC)
                self.assertIn("Groq", result["provider"])
                self.assertEqual(call.call_count, 2)
                self.assertEqual(call.call_args.kwargs["output_tokens"], 1100)

    def test_gemini_404_then_groq(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "", "GEMINI_API_KEY": "fake", "GROQ_API_KEY": "fake"}):
            with patch.object(v3, "gemini_models", return_value=["gemini-2.5-flash-lite"]), \
                    patch.object(v3, "gemini_call", side_effect=v3.ProviderError(404, "model not available")), \
                    patch.object(v3, "openai_call", return_value=ANSWER):
                result = v3.generate({}, PAPERS, NEWS, TOOLS, VIDEOS, SEC)
                self.assertIn("Groq", result["provider"])

    def test_source_fallback_if_no_keys(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "", "GEMINI_API_KEY": "", "GROQ_API_KEY": "", "OLLAMA_MODEL": ""}):
            result = v3.generate({}, PAPERS, NEWS, TOOLS, VIDEOS, SEC)
            self.assertEqual(result["provider"], "Source-only fallback")

    def test_hn_uses_api_and_time_filter(self):
        now = int(v3.datetime.now(v3.timezone.utc).timestamp())
        with patch.object(v3, "request_json", return_value={"hits": [{
            "objectID": "12", "title": "AI tools", "url": "https://example.org/h", "created_at_i": now}]}) as req:
            entries = v3.hn_items([], 5, 24, "general-ai-news", False)
            self.assertEqual(len(entries), 1)
            self.assertIn("hn.algolia.com", req.call_args.args[1])

    def test_transcripts_off_by_default(self):
        with patch.dict(os.environ, {"ENABLE_YOUTUBE_TRANSCRIPTS": "false"}):
            text, available = v3.transcript_excerpt(None, "video")
            self.assertFalse(available)
            self.assertIn("title", text)


if __name__ == "__main__":
    unittest.main()
