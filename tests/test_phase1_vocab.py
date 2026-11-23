"""Phase 1, Workstream 2: one shared vocabulary normalizer.

Acceptance: the same word is tokenized identically by Brain tags,
Attention keyword extraction and Goal keywords; relation aliases are
canonicalized without silently merging different relations; the
capital_of / capital_of_country mismatch is fixed.
"""
import unittest

from ab.brain import Brain, _tags
from ab.core.attention import Attention
from ab.core.thought import Thought
from ab.core.memory import LongTermMemory
from ab.core.vocab import FUNCTIONAL_RELATIONS, normalize_relation, normalize_text, tokenize
from ab.planning.goals import Goal


class TestNormalize(unittest.TestCase):
    def test_normalize_text(self):
        self.assertEqual(normalize_text("  The Cat. "), "cat")
        self.assertEqual(normalize_text("PARIS"), "paris")
        self.assertEqual(normalize_text("new   york"), "new york")
        self.assertEqual(normalize_text("a"), "a")

    def test_tokenize_is_deterministic_and_capped(self):
        # "the" is 3 chars, so it is kept; "is"/"in" are 2 chars, dropped.
        self.assertEqual(tokenize("The key is in the lab"), {"the", "key", "lab"})
        many = " ".join(f"word{i}" for i in range(30))
        self.assertLessEqual(len(tokenize(many)), 12)

    def test_relation_aliases(self):
        self.assertEqual(normalize_relation("capital_of_country"), "capital_of")
        self.assertEqual(normalize_relation("Capital City"), "capital_of")
        self.assertEqual(normalize_relation("capital"), "capital_of")
        # Semantically different relations are NOT merged.
        self.assertEqual(normalize_relation("is_a"), "is_a")
        self.assertEqual(normalize_relation("boils_at"), "boils_at")
        self.assertEqual(normalize_relation("located_in"), "located_in")

    def test_functional_relations_fixed(self):
        self.assertIn("capital_of", FUNCTIONAL_RELATIONS)
        self.assertNotIn("capital_of_country", FUNCTIONAL_RELATIONS)


class TestOneNormalizerEverywhere(unittest.TestCase):
    def test_brain_attention_goal_agree_on_words(self):
        text = "the key is in the lab"
        tags = _tags(text)
        self.assertEqual(tags, tokenize(text))
        thought_words = Attention._words(Thought(text, "percept"))
        self.assertEqual(thought_words, tags)
        goal_words = Goal("the key is in the lab", {}).keywords
        self.assertTrue(tags <= goal_words)

    def test_capital_of_conflict_is_now_disputed(self):
        # v0.1: functional list said "capital_of_country", so this conflict
        # was never flagged. Phase 1: it is.
        m = LongTermMemory()
        m.add("paris", "capital_of", "france", "teacher")
        result, new = m.add("paris", "capital_of", "london", "web:x.org")
        self.assertEqual(new.status, "disputed")

    def test_non_functional_conflict_is_not_disputed(self):
        m = LongTermMemory()
        m.add("cat", "is_a", "mammal", "teacher")
        result, new = m.add("cat", "is_a", "robot", "web:x.org")
        self.assertNotEqual(new.status, "disputed")

    def test_ask_is_case_and_alias_insensitive(self):
        b = Brain()
        b.teach_fact("paris", "capital_of", "france")
        self.assertEqual(b.ask("PARIS", "Capital_Of").best[0], "france")
        self.assertEqual(b.ask("paris", "capital_of_country").best[0], "france")


if __name__ == "__main__":
    unittest.main()
