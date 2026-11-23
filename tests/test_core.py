import unittest
from ab.core.thought import Thought
from ab.core.working_memory import WorkingMemory
from ab.core.attention import Attention
from ab.core.control import CognitiveControl


class TestWorkingMemory(unittest.TestCase):
    def test_capacity_never_exceeded_and_weakest_displaced(self):
        wm = WorkingMemory(capacity=3)
        for i, s in enumerate([0.9, 0.5, 0.7]):
            t = Thought(f"t{i}"); t.salience = s; wm.admit(t)
        strong = Thought("strong"); strong.salience = 0.8
        out = wm.admit(strong)
        self.assertEqual(len(wm), 3)
        self.assertEqual(out[0].content, "t1")            # 0.5 was weakest
        weak = Thought("weak"); weak.salience = 0.1
        self.assertEqual(wm.admit(weak)[0].content, "weak")   # rejected, not admitted
        self.assertEqual(len(wm), 3)

    def test_decay_fades_thoughts(self):
        wm = WorkingMemory(capacity=3, decay=0.5, floor=0.1)
        t = Thought("x"); t.salience = 0.3; wm.admit(t)
        faded = []
        for _ in range(4):
            faded += wm.tick()
        self.assertEqual(len(wm), 0)
        self.assertEqual(len(faded), 1)


class TestAttention(unittest.TestCase):
    def test_goal_relevant_and_urgent_outrank_noise(self):
        a = Attention()
        relevant = Thought("the key is in the lab", "percept", urgency=0.5, tags={"key"})
        noise = Thought("weather is nice", "percept", urgency=0.1)
        top = a.focus([noise, relevant], {"key", "lab"}, cycle=1, k=1)
        self.assertEqual(top[0], relevant)

    def test_repeated_thought_loses_novelty(self):
        a = Attention()
        t1 = Thought("same", "percept"); t2 = Thought("same", "percept")
        s1 = a.score(t1, set(), 0); a.focus([t1], set(), 0)
        self.assertLess(a.score(t2, set(), 0), s1)

    def test_interrupt_threshold(self):
        self.assertTrue(Attention().interrupts(Thought("fire", urgency=0.95)))
        self.assertFalse(Attention().interrupts(Thought("meh", urgency=0.4)))


class TestControl(unittest.TestCase):
    def test_hysteresis(self):
        c = CognitiveControl(switch_margin=0.2)
        self.assertFalse(c.should_switch(0.5, 0.6))
        self.assertTrue(c.should_switch(0.5, 0.8))

    def test_inhibition(self):
        c = CognitiveControl(min_confidence=0.3, inhibited_tags=["spam"])
        kept, blocked = c.inhibit([Thought("ok"), Thought("doubt", confidence=0.1), Thought("s", tags={"spam"})])
        self.assertEqual(len(kept), 1); self.assertEqual(len(blocked), 2)

    def test_stall_and_budget(self):
        c = CognitiveControl(stall_window=3, max_cycles=10)
        self.assertTrue(c.is_stalled(["a", "a", "a"]))
        self.assertFalse(c.is_stalled(["a", "b", "a"]))
        self.assertTrue(c.over_budget(10))

    def test_conflict_resolution(self):
        out = CognitiveControl.resolve_conflicts([("x", 0.9, frozenset({"hand"})),
                                                  ("y", 0.5, frozenset({"hand"})),
                                                  ("z", 0.4, frozenset({"eyes"}))])
        self.assertEqual(out, ["x", "z"])


if __name__ == "__main__":
    unittest.main()
