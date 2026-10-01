import contextlib
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import transcribe


def transcript(text="Example text."):
    return SimpleNamespace(
        status=transcribe.aai.TranscriptStatus.completed,
        text=text,
        utterances=[SimpleNamespace(speaker="A", text=text)],
        speech_understanding=None,
    )


def load_entry_point(filename):
    path = Path(__file__).resolve().parents[1] / "src" / filename
    spec = importlib.util.spec_from_file_location("legacy_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "outputs"
        service = patch.object(transcribe.aai, "Transcriber")
        self.client = service.start().return_value
        self.addCleanup(service.stop)
        self.client.transcribe.return_value = transcript()

    def audio(self, name="sample.wav"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic fixture; service is mocked")
        return str(path)

    def run_cli(self, args, entry=transcribe.main):
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr):
            try:
                entry(args)
                return 0
            except SystemExit as exc:
                return exc.code

    def test_single_file_keeps_output_and_model_settings(self):
        source = self.audio()
        code = self.run_cli([source, "--output-dir", str(self.output), "--disfluencies", "--language-detection"])
        self.assertEqual(code, 0)
        self.assertEqual((self.output / "sample.txt").read_text(), "Example text.")
        config = self.client.transcribe.call_args.kwargs["config"]
        self.assertEqual(config.speech_models, transcribe.ASYNC_SPEECH_MODELS)
        self.assertTrue(config.disfluencies)
        self.assertTrue(config.language_detection)

    def test_mixed_folder_file_and_url_skip_non_audio_and_subfolders(self):
        self.audio("folder/second.MP4")
        self.audio("folder/first.wav")
        self.audio("folder/notes.txt")
        self.audio("folder/nested/hidden.wav")
        single = self.audio("single.wav")
        url = "https://example.com/remote%20audio.mp3?download=1"
        code = self.run_cli([str(self.root / "folder"), single, url, "--output-dir", str(self.output)])
        self.assertEqual(code, 0)
        self.assertEqual({p.name for p in self.output.iterdir()}, {"first.txt", "second.txt", "single.txt", "remote audio.txt"})
        sources = {call.args[0] for call in self.client.transcribe.call_args_list}
        self.assertIn(url, sources)
        self.assertEqual(len(sources), 4)

    def test_duplicate_destinations_are_rejected_before_service_calls(self):
        for inputs in [
            [self.audio("one/same.wav"), self.audio("two/SAME.mp3")],
            [self.audio("same.wav"), "https://example.com/same.mp3?download=1"],
        ]:
            with self.subTest(inputs=inputs):
                self.assertEqual(self.run_cli([*inputs, "--output-dir", str(self.output)]), 2)
        self.client.transcribe.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_output_cannot_overwrite_an_input(self):
        source = self.audio()
        original = Path(source).read_bytes()
        self.assertEqual(self.run_cli([source, "-o", source]), 2)
        self.assertEqual(Path(source).read_bytes(), original)
        self.client.transcribe.assert_not_called()

    def test_missing_input_does_not_stop_valid_inputs(self):
        source = self.audio()
        self.assertEqual(self.run_cli([str(self.root / "missing.wav"), source, "--output-dir", str(self.output)]), 1)
        self.assertTrue((self.output / "sample.txt").exists())
        self.client.transcribe.assert_called_once()

    def test_failed_service_does_not_stop_other_inputs(self):
        good, bad = self.audio("good.wav"), self.audio("bad.wav")

        def result(path, **kwargs):
            if path == bad:
                return SimpleNamespace(status=transcribe.aai.TranscriptStatus.error, error="mock service failure")
            return transcript()

        self.client.transcribe.side_effect = result
        self.assertEqual(self.run_cli([bad, good, "--output-dir", str(self.output)]), 1)
        self.assertTrue((self.output / "good.txt").exists())
        self.assertFalse((self.output / "bad.txt").exists())

    def test_empty_and_unreadable_folders_exit_without_service_calls(self):
        folder = self.root / "empty"
        folder.mkdir()
        self.assertEqual(self.run_cli([str(folder)]), 1)
        with patch.object(transcribe.os, "listdir", side_effect=PermissionError):
            self.assertEqual(self.run_cli([str(folder)]), 1)
        self.client.transcribe.assert_not_called()

    def test_output_options_and_invalid_counts_fail_before_service_calls(self):
        first, second = self.audio("first.wav"), self.audio("second.wav")
        invalid = [
            [first, second, "-o", str(self.output)],
            [first, "-o", str(self.output), "--output-dir", str(self.root)],
            [first, "--jobs", "0"],
            [first, "--speakers", "0"],
            [first, "--speaker-names", ""],
            [first, "--speaker-roles", "a" * 36],
            [first, "--speaker-names", "Alex", "--speaker-roles", "host"],
        ]
        for args in invalid:
            with self.subTest(args=args):
                self.assertEqual(self.run_cli(args), 2)
        self.client.transcribe.assert_not_called()

    def test_parallel_jobs_are_bounded_and_all_outputs_saved(self):
        paths = [self.audio(f"clip{i}.wav") for i in range(4)]
        barrier = threading.Barrier(2, timeout=5)
        lock = threading.Lock()
        active = 0
        peak = 0

        def result(*args, **kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            barrier.wait()
            with lock:
                active -= 1
            return transcript()

        self.client.transcribe.side_effect = result
        self.assertEqual(self.run_cli([*paths, "--jobs", "2", "--output-dir", str(self.output)]), 0)
        self.assertEqual(peak, 2)
        self.assertEqual(len(list(self.output.iterdir())), 4)

    def test_speaker_aliases_and_expected_count_reach_sdk_constructor(self):
        source = self.audio()
        self.assertEqual(self.run_cli([source, "-d", "-s", "2", "--output-dir", str(self.output)]), 0)
        config = self.client.transcribe.call_args.kwargs["config"]
        self.assertTrue(config.speaker_labels)
        self.assertEqual(config.speakers_expected, 2)
        self.assertEqual((self.output / "sample.txt").read_text(), "Speaker A: Example text.")

    def test_speaker_identification_retains_names_and_roles(self):
        source = self.audio()
        for option, known, speaker_type in [
            ("--speaker-names", "Alex,Sam", "name"),
            ("--speaker-roles", "host,guest", "role"),
        ]:
            with self.subTest(option=option):
                value = transcript()
                value.speech_understanding = SimpleNamespace(response=SimpleNamespace(
                    speaker_identification=SimpleNamespace(mapping={"A": known.split(",")[0]})
                ))
                self.client.transcribe.return_value = value
                self.assertEqual(self.run_cli([source, option, known, "--output-dir", str(self.output)]), 0)
                config = self.client.transcribe.call_args.kwargs["config"]
                identification = config.speech_understanding.request.speaker_identification
                self.assertEqual(identification.known_values, known.split(","))
                self.assertEqual(identification.speaker_type, speaker_type)
                self.assertEqual((self.output / "sample.txt").read_text(), known.split(",")[0] + ": Example text.")

    def test_legacy_entry_point_retains_output_naming(self):
        legacy = load_entry_point("main.py")
        source = self.audio()
        with patch.object(legacy.os, "getcwd", return_value=str(self.root)):
            self.assertEqual(self.run_cli([source, "--diarize", "--speakers", "2"], entry=legacy.main), 0)
        output = self.root / "transcripts" / "sample_transcription.txt"
        self.assertEqual(output.read_text(), "Speaker A: Example text.")

    def test_legacy_stream_entry_point_preserves_blackhole_and_allows_override(self):
        legacy = load_entry_point("transcript_stream.py")
        with patch.object(legacy, "live_main") as main:
            legacy.main(["--no-save"])
            main.assert_called_with(["--device", "BlackHole", "--no-save"])
            legacy.main(["--device=2"])
            main.assert_called_with(["--device=2"])


class ConfigurationTests(unittest.TestCase):
    def load_config(self):
        path = Path(transcribe.__file__).with_name("config.py")
        spec = importlib.util.spec_from_file_location("test_config", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_environment_priority_and_legacy_fallback_without_reading_env_file(self):
        for environment, expected in [
            ({"ASSEMBLYAI_API_KEY": "offline-test-key", "API_KEY": "legacy-test-key"}, "offline-test-key"),
            ({"API_KEY": "legacy-test-key"}, "legacy-test-key"),
        ]:
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True), patch("dotenv.load_dotenv") as load:
                self.assertEqual(self.load_config().ASSEMBLYAI_API_KEY, expected)
                load.assert_not_called()

    def test_dotenv_path_is_project_local(self):
        with patch.dict(os.environ, {}, clear=True), patch("dotenv.load_dotenv") as load:
            self.load_config()
            load.assert_called_once_with(Path(transcribe.__file__).with_name(".env"))


if __name__ == "__main__":
    unittest.main()
