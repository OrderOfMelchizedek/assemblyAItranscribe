import contextlib
import io
import unittest
from unittest.mock import patch

import live
from assemblyai.streaming.v3.models import StreamingEvents


def turn(order, text="Hello.", speaker="A", final=True):
    return live.RevisionCompatibleClient._parse_message({
        "type": "Turn", "turn_order": order, "turn_is_formatted": True,
        "end_of_turn": final, "transcript": text, "end_of_turn_confidence": 1.0,
        "speaker_label": speaker,
        "words": [{"text": text, "start": order * 100, "end": order * 100 + 90,
                   "confidence": 0.9, "word_is_final": final, "speaker": speaker}],
    })


def revision(order, speaker):
    # Use the documented wire shape, without confidence or word_is_final.
    return live.RevisionCompatibleClient._parse_message({
        "type": "SpeakerRevision", "revisions": [{
            "turn_order": order, "speaker_label": speaker,
            "words": [{"text": "Hello.", "start": order * 100,
                       "end": order * 100 + 90, "speaker": speaker}],
        }],
    })


class LiveTranscriptTests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def test_revisions_replace_labels_and_preserve_text_and_order(self):
        transcriber = live.LiveTranscriber(speaker_labels=True)
        transcriber.on_turn(None, turn(2, "Second."))
        transcriber.on_turn(None, turn(1))
        transcriber.on_speaker_revision(None, revision(1, "B"))
        self.assertEqual(transcriber.get_full_transcript(), "Speaker B: Hello.\nSpeaker A: Second.")
        self.assertEqual(transcriber.turns_by_order[1].words[0].speaker, "B")
        self.assertEqual(transcriber.turns_by_order[1].words[0].text, "Hello.")
        transcriber.on_speaker_revision(None, revision(1, "C"))
        transcriber.on_speaker_revision(None, revision(99, "B"))
        self.assertEqual(transcriber.get_full_transcript(), "Speaker C: Hello.\nSpeaker A: Second.")
        transcriber.on_speaker_revision(None, revision(1, None))
        self.assertEqual(transcriber.get_full_transcript(), "Hello.\nSpeaker A: Second.")

    def test_partials_and_duplicate_finals_do_not_duplicate_saved_text(self):
        transcriber = live.LiveTranscriber()
        transcriber.on_turn(None, turn(0, "Hel", final=False))
        self.assertEqual(transcriber.get_full_transcript(), "")
        transcriber.on_turn(None, turn(0, "hello"))
        transcriber.on_turn(None, turn(0, "Hello."))
        transcriber.on_turn(None, turn(1, "Goodbye."))
        self.assertEqual(transcriber.get_full_transcript(), "Hello. Goodbye.")

    def test_shutdown_applies_final_revision_and_requests_periodic_updates(self):
        for labels in (False, True):
            with self.subTest(labels=labels):
                transcriber = live.LiveTranscriber(speaker_labels=labels)
                handlers = {}
                with patch.object(live, "RevisionCompatibleClient") as client_class, \
                        patch.object(live.sd, "InputStream"), \
                        patch.object(live.sd, "sleep", side_effect=KeyboardInterrupt):
                    client = client_class.return_value
                    client.on.side_effect = lambda event, handler: handlers.update({event: handler})
                    client.connect.side_effect = lambda params: handlers[StreamingEvents.Turn](client, turn(0))
                    client.disconnect.side_effect = lambda **kwargs: handlers[StreamingEvents.SpeakerRevision](client, revision(0, "B"))
                    # Parsing fixtures must use the real client, not this mock.
                    client_class._parse_message.side_effect = ORIGINAL_PARSE
                    transcriber.start(0)
                    params = client.connect.call_args.args[0].model_dump(mode="json", exclude_none=True)
                    self.assertEqual(params["speech_model"], "universal-3-6-pro")
                    self.assertEqual(params.get("speaker_labels_revision_interval_ms"), 300000 if labels else None)
                    client.disconnect.assert_called_once_with(terminate=True)
                self.assertEqual(transcriber.get_full_transcript(), "Speaker B: Hello." if labels else "Hello.")


ORIGINAL_PARSE = live.RevisionCompatibleClient._parse_message


if __name__ == "__main__":
    unittest.main()
