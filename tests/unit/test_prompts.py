from prompts import SYSTEM_PROMPT


class TestSystemPrompt:
    def test_is_non_empty_string(self):
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 0

    def test_contains_jp_morgan(self):
        assert "J.P. Morgan" in SYSTEM_PROMPT

    def test_contains_mid_year_outlook(self):
        assert "Mid-Year Outlook" in SYSTEM_PROMPT

    def test_contains_instructions(self):
        assert "don't know" in SYSTEM_PROMPT.lower() or "not in the context" in SYSTEM_PROMPT.lower()
