import unittest

import runway_adapter
import story_board_maker


class MultilingualStoryboardTests(unittest.TestCase):
    def test_sentence_boundaries_across_scripts(self):
        samples = {
            "English": "The bucket fell. He looked at his hands. He felt old.",
            "Bengali": "বালতি পড়ে গেল। তিনি হাতের দিকে তাকালেন। তিনি বুড়ো হয়েছেন।",
            "Hindi": "बाल्टी गिर गई। उसने अपने हाथों को देखा। वह बूढ़ा था।",
            "Arabic": "سقط الدلو. نظر إلى يديه؟ لقد أصبح عجوزاً.",
            "Chinese": "水桶掉进井里。他看着双手。他老了。",
            "Japanese": "バケツが落ちた。彼は手を見た。彼は老いた。",
        }
        for language, text in samples.items():
            with self.subTest(language=language):
                self.assertEqual(len(story_board_maker._split_sentences(text)), 3)

    def test_combining_mark_scripts_count_words_not_codepoints(self):
        bengali = "হেমকান্তের হাত থেকে বালতি পড়ে গেল।"
        hindi = "बाल्टी कुएँ में गिर गई।"
        self.assertEqual(story_board_maker.word_count(bengali), 6)
        self.assertEqual(story_board_maker.word_count(hindi), 5)

    def test_no_space_scripts_receive_speech_unit_estimate(self):
        self.assertGreater(story_board_maker.word_count("水桶掉进了井里。"), 1)
        self.assertGreater(story_board_maker.word_count("バケツが井戸に落ちた。"), 1)

    def test_runway_segments_use_ordered_action_beats(self):
        actions = [
            {
                "description": (
                    "The bucket slips into the well. Then the man studies his hands. "
                    "Finally he turns away without retrieving it."
                )
            }
        ]
        prompts = [
            runway_adapter._segment_action_text(actions, index, 3)
            for index in range(1, 4)
        ]
        self.assertEqual(len(set(prompts)), 3)
        self.assertIn("bucket", prompts[0])
        self.assertIn("hands", prompts[1])
        self.assertIn("turns away", prompts[2])


if __name__ == "__main__":
    unittest.main()
