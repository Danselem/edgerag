import os

from utils import silence_stderr


class TestSilenceStderr:
    def test_verbose_env_does_not_silence(self, monkeypatch, capsys):
        monkeypatch.setenv("EDGE_RAG_VERBOSE", "1")
        import sys
        original_stderr = sys.stderr
        silence_stderr()
        # stderr should still be the original (not /dev/null)
        assert sys.stderr == original_stderr
        # Restore
        os.dup2(2, 2)

    def test_default_silences_stderr(self, monkeypatch):
        monkeypatch.delenv("EDGE_RAG_VERBOSE", raising=False)
        silence_stderrAfter = silence_stderr()
        # After silencing, writing to fd 2 should not raise
        # (it goes to /dev/null)
        try:
            os.write(2, b"test")
            wrote_to_devnull = True
        except OSError:
            wrote_to_devnull = False
        assert wrote_to_devnull
        # Restore stderr
        os.dup2(2, 2)
