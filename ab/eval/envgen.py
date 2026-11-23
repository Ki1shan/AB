"""Novel environment generator (Phase 9, design SS3.3).

A seeded generator producing FRESH environments of the existing
families (treasure-like multi-room, workshop-like tool worlds,
surprise-like hidden-rule worlds) with different names, layouts and
object counts per seed. ``random.Random(seed)``; stdlib only; no LLM.

``generate(seed, family)`` returns ``(env, actions, goal,
expected_solution_shape)``. The brain never sees the generator's
seed; the consolidated runner records the seeds it used. Fixed
seeds are deterministic: the same seed always yields the same names,
layout and object counts (verified in ``tests/test_phase9_reasoning2.py``).

Every generated world is solvable BY CONSTRUCTION (a chain layout
with the key before the door, the mould before the press, the coil
warmer before the heat step), never by luck: the expected solution
shape is derived from the construction, not searched for afterwards.
Names come from invented pools that appear nowhere in ``ab/``
outside ``ab/eval/``.
"""
from __future__ import annotations

import random

from ..body.environment import Environment
from ..planning.planner import Action

FAMILIES = ("treasure", "workshop", "surprise")

# Invented name pools (eval-only names; none appears in ``ab/``
# outside ``ab/eval/``).
_ROOMS = ("vortan", "drelm", "forvik", "trellin", "vossar", "quivern",
          "norbex", "mellor", "krelm", "garnik", "zarn", "phel", "orin",
          "talvik", "senvik", "mordel", "vexar", "lomin", "dravik", "senn")
_KEYS = ("krelm", "vexar", "lomin", "talvik", "orin", "phel")
_TREASURES = ("garnik", "senn", "dravik", "mordel", "zarn", "senvik")
_TOOLS = ("cutter", "shears", "snips", "awl", "reamer")
_MATERIALS = ("copper", "brass", "steel", "tin", "lead", "zinc")
_MOULDS = ("mould", "cast", "die", "form")
_SUBSTANCES = ("vapour", "resin", "flux", "alloy")

# Phase 10 (P10-2) hard-mode pools. New invented names, eval-only
# (see ``docs/phase10-design.md`` §P10-2); add them to the leakage
# audit with the other envgen names.
_DECOYS = ("thrum", "clack", "toll")


# ------------------------------------------------------------------ treasure-like
class GeneratedTreasureWorld(Environment):
    """Multi-room chain: the key waits in the second room, the door
    gates the last room, the treasure waits there. ``spec`` carries
    the room chain, key/treasure names and their state keys."""

    def __init__(self, spec: dict):
        self.spec = spec
        rooms = spec["rooms"]
        self.adj = {r: [] for r in rooms}
        for a, b in zip(rooms, rooms[1:]):
            self.adj[a].append(b)
            self.adj[b].append(a)
        self.s = {"at": rooms[0], spec["key_state"]: False,
                  "door_open": False, spec["treasure_state"]: False}
        self.log: list[str] = []

    def observe(self) -> dict:
        return dict(self.s)

    def act(self, name: str, args: dict) -> tuple[bool, str]:
        spec, s = self.spec, self.s
        rooms = spec["rooms"]
        ok, info = False, "nothing happened"
        if name == "go":
            dest = args.get("room")
            if dest in self.adj[s["at"]] and (
                    dest != rooms[-1] or s["door_open"]):
                s["at"], ok, info = dest, True, f"moved to {dest}"
            else:
                info = f"cannot reach {dest}"
        elif name == f"take_{spec['key']}" and s["at"] == rooms[1] \
                and not s[spec["key_state"]]:
            s[spec["key_state"]], ok, info = True, True, f"took the {spec['key']}"
        elif name == "unlock" and s["at"] == rooms[1] and s[spec["key_state"]]:
            s["door_open"], ok, info = True, True, "unlocked the door"
        elif name == f"take_{spec['treasure']}" and s["at"] == rooms[-1]:
            s[spec["treasure_state"]], ok, info = True, True, f"took the {spec['treasure']}"
        self.log.append(f"{name}{args or ''}: {info}")
        return ok, info


def _treasure_actions(spec: dict) -> list[Action]:
    rooms = spec["rooms"]
    actions = []
    for a, b in zip(rooms, rooms[1:]):
        # The last hop is gated: the believed model knows the door
        # must be open (the env enforces the same gate), so the plan
        # must take the key and unlock first.
        pre = {"at": a, "door_open": True} if b == rooms[-1] else {"at": a}
        actions.append(Action(f"go_{b}_from_{a}", pre, {"at": b},
                              args={"_do": "go", "room": b}))
        actions.append(Action(f"go_{a}_from_{b}", {"at": b}, {"at": a},
                              args={"_do": "go", "room": a}))
    # The door gate is on the last hop; the return hop needs no key.
    actions.append(Action(f"take_{spec['key']}",
                          {"at": rooms[1], spec["key_state"]: False},
                          {spec["key_state"]: True},
                          args={"_do": f"take_{spec['key']}"}))
    actions.append(Action("unlock",
                          {"at": rooms[1], spec["key_state"]: True},
                          {"door_open": True}, args={"_do": "unlock"}))
    actions.append(Action(f"take_{spec['treasure']}",
                          {"at": rooms[-1]},
                          {spec["treasure_state"]: True},
                          args={"_do": f"take_{spec['treasure']}"}))
    return actions


# ------------------------------------------------------------------ workshop-like
class GeneratedWorkshopWorld(Environment):
    """Tool world: the durable tool cuts one material per room, the
    press needs the mould seated, assembly in the last room makes the
    treasure. The mould waits in the last room; the press has a cheap
    variant (no mould, fails) and an expensive one (mould seated,
    works). Cut/press are environment actions here (no registry
    tool), so the generated world runs without fork isolation."""

    def __init__(self, spec: dict):
        self.spec = spec
        rooms = spec["rooms"]
        self.adj = {r: [] for r in rooms}
        for a, b in zip(rooms, rooms[1:]):
            self.adj[a].append(b)
            self.adj[b].append(a)
        self.s = {"at": rooms[0], spec["tool_key"]: False,
                  spec["mould_key"]: False, "door_open": False,
                  "pressed": False, spec["treasure_key"]: False}
        self.s.update({ck: False for _m, ck in spec["cuts"]})
        self.log: list[str] = []

    def observe(self) -> dict:
        return dict(self.s)

    def act(self, name: str, args: dict) -> tuple[bool, str]:
        spec, s = self.spec, self.s
        rooms = spec["rooms"]
        ok, info = False, "nothing happened"
        if name == "go":
            dest = args.get("room")
            if dest in self.adj[s["at"]] and (
                    dest != rooms[-1] or s["door_open"]):
                s["at"], ok, info = dest, True, f"moved to {dest}"
            else:
                info = f"cannot reach {dest}"
        elif name == f"take_{spec['tool']}" and s["at"] == rooms[1] \
                and not s[spec["tool_key"]]:
            s[spec["tool_key"]], ok, info = True, True, f"took the {spec['tool']}"
        elif name == f"take_{spec['mould']}" and s["at"] == rooms[-1] \
                and not s[spec["mould_key"]]:
            s[spec["mould_key"]], ok, info = True, True, f"took the {spec['mould']}"
        elif name == "cut":
            material = args.get("material")
            cut_key = next((ck for m, ck in spec["cuts"] if m == material),
                           None)
            if cut_key is not None and s[spec["tool_key"]] and not s[cut_key]:
                s[cut_key], ok, info = True, True, f"cut the {material}"
            else:
                info = "nothing to cut here"
        elif name == "pry_gate" and s["at"] == rooms[-2] and s[spec["tool_key"]]:
            s["door_open"], ok, info = True, True, "pried the gate open"
        elif name == "press" and s[spec["mould_key"]]:
            s["pressed"], ok, info = True, True, "the press stamps the parts"
        elif name == "press":
            info = f"the press has no {spec['mould']} seated"
        elif name == "assemble" and s["at"] == rooms[-1] and s["pressed"] \
                and all(s[ck] for _m, ck in spec["cuts"]):
            s[spec["treasure_key"]], ok, info = True, True, f"assembled the {spec['treasure']}"
        elif name == "assemble":
            info = "parts missing for assembly"
        self.log.append(f"{name}{args or ''}: {info}")
        return ok, info


def _workshop_actions(spec: dict) -> list[Action]:
    rooms = spec["rooms"]
    tool, mould = spec["tool"], spec["mould"]
    tk, mk = spec["tool_key"], spec["mould_key"]
    actions = []
    for a, b in zip(rooms, rooms[1:]):
        gated = (b == rooms[-1])
        pre = {"at": a, "door_open": True} if gated else {"at": a}
        actions.append(Action(f"go_{b}_from_{a}", pre, {"at": b},
                              args={"_do": "go", "room": b}))
        actions.append(Action(f"go_{a}_from_{b}", {"at": b}, {"at": a},
                              args={"_do": "go", "room": a}))
    actions.append(Action(f"take_{tool}", {"at": rooms[1], tk: False},
                          {tk: True}, args={"_do": f"take_{tool}"}))
    actions.append(Action(f"take_{mould}", {"at": rooms[-1], mk: False},
                          {mk: True}, args={"_do": f"take_{mould}"}))
    for (material, cut_key), room in zip(spec["cuts"], rooms[1:]):
        actions.append(Action(
            "cut", {"at": room, tk: True}, {cut_key: True}, cost=1.0,
            args={"material": material}))
    actions.append(Action("pry_gate", {"at": rooms[-2], tk: True},
                          {"door_open": True}, cost=5.0,
                          args={"_do": "pry_gate"}))
    actions.append(Action("press", {"at": rooms[1]}, {"pressed": True},
                          cost=0.3, args={"_do": "press"}))
    actions.append(Action("press", {"at": rooms[-1], mk: True},
                          {"pressed": True}, cost=2.0, args={"_do": "press"}))
    actions.append(Action(
        "assemble",
        {"at": rooms[-1], **{ck: True for _m, ck in spec["cuts"]},
         "pressed": True},
        {spec["treasure_key"]: True}, args={"_do": "assemble"}))
    return actions


# ------------------------------------------------------------------ surprise-like
class GeneratedSurpriseWorld(Environment):
    """Hidden-rule world: the ``heat`` action is believed to make the
    substance hot, but with a cold coil it stays cold. Warming the
    coil first makes the same action succeed."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.s = {"at": spec["rooms"][0], "coil": "cold",
                  spec["hot_key"]: False}
        self.log: list[str] = []

    def observe(self) -> dict:
        return dict(self.s)

    def act(self, name: str, args: dict) -> tuple[bool, str]:
        s = self.s
        ok, info = False, "nothing happened"
        if name == "warm_coil" and s["coil"] == "cold":
            s["coil"], ok, info = "warm", True, "the coil is warm now"
        elif name == f"heat_{self.spec['substance']}" and s["coil"] == "warm":
            s[self.spec["hot_key"]], ok, info = True, True, (
                f"the {self.spec['substance']} is hot now")
        elif name == f"heat_{self.spec['substance']}":
            info = "the coil hums but the substance stays cold"
        self.log.append(f"{name}: {info}")
        return ok, info


def _surprise_actions(spec: dict) -> list[Action]:
    return [
        Action(f"heat_{spec['substance']}", {"at": spec["rooms"][0]},
               {spec["hot_key"]: True}, cost=1.0,
               args={"_do": f"heat_{spec['substance']}"}),
        Action("warm_coil", {"at": spec["rooms"][0], "coil": "cold"},
               {"coil": "warm"}, cost=1.0, args={"_do": "warm_coil"}),
    ]


# ------------------------------------------------------------------ hard mode (Phase 10, P10-2)
# Harder generated worlds. These are SEPARATE classes from the
# baseline ones above: ``generate(seed, family)`` with no difficulty
# argument keeps the exact baseline behaviour (same specs, shapes and
# rng consumption), and the hard builders run only when a difficulty
# is passed. Every hard world is solvable by construction, like the
# baseline ones; the difficulty is recorded in the shape under
# ``"difficulty"`` (knobs + derived dimensions) and never tuned to
# flatter the brain -- the numbers are reported either way.
HARD_DIFFICULTY = {"treasure": {"n_rooms": 5, "second_gate": True,
                                "decoy": True},
                   "workshop": {"decoy_variant": True,
                                "breakable_tool": True},
                   "surprise": {"two_stage": True}}


def _resolve_difficulty(family: str, difficulty) -> dict:
    """Normalise a difficulty argument to a knob dict. ``None`` ->
    no hard mode (the caller must not call the hard builders);
    ``"hard"`` -> the family's full hard knobs; a dict -> its keys
    over the family's hard defaults (explicit ``False`` disables a
    knob). Unknown keys raise ``ValueError`` (a typo must not
    silently build an easier world)."""
    known = {"treasure": {"n_rooms", "second_gate", "decoy"},
             "workshop": {"decoy_variant", "breakable_tool"},
             "surprise": {"two_stage"}}
    if difficulty is None:
        return {}
    if isinstance(difficulty, str):
        if difficulty != "hard":
            raise ValueError(f"unknown envgen difficulty: {difficulty!r}")
        return dict(HARD_DIFFICULTY[family])
    if isinstance(difficulty, dict):
        unknown = sorted(set(difficulty) - known[family])
        if unknown:
            raise ValueError(f"unknown envgen difficulty knobs for "
                             f"{family}: {unknown}")
        out = dict(HARD_DIFFICULTY[family])
        out.update(difficulty)
        return out
    raise ValueError(f"envgen difficulty must be None, 'hard' or a dict, "
                     f"got {type(difficulty).__name__}")


class HardTreasureWorld(Environment):
    """Hard treasure: a 5-room chain (default) with TWO gated doors,
    each needing its own key, plus a decoy tool-less opener for the
    second door (cheap, always fails -- the P-A shape). Key A waits
    in room 2, key B behind door A; the treasure waits behind door B.
    ``spec`` carries the chain, both keys/treasure and the decoy."""

    def __init__(self, spec: dict):
        self.spec = spec
        rooms = spec["rooms"]
        self.adj = {r: [] for r in rooms}
        for a, b in zip(rooms, rooms[1:]):
            self.adj[a].append(b)
            self.adj[b].append(a)
        self.s = {"at": rooms[0], spec["key_a_state"]: False,
                  spec["key_b_state"]: False, "door_a_open": False,
                  "door_b_open": False, spec["treasure_state"]: False}
        self.log: list[str] = []

    def observe(self) -> dict:
        return dict(self.s)

    def act(self, name: str, args: dict) -> tuple[bool, str]:
        spec, s = self.spec, self.s
        rooms = spec["rooms"]
        ok, info = False, "nothing happened"
        if name == "go":
            dest = args.get("room")
            gated_a = dest == rooms[-2] and not s["door_a_open"]
            gated_b = dest == rooms[-1] and not s["door_b_open"]
            if dest in self.adj[s["at"]] and not gated_a and not gated_b:
                s["at"], ok, info = dest, True, f"moved to {dest}"
            else:
                info = f"cannot reach {dest}"
        elif name == f"take_{spec['key_a']}" and s["at"] == rooms[1] \
                and not s[spec["key_a_state"]]:
            s[spec["key_a_state"]], ok, info = True, True, f"took the {spec['key_a']}"
        elif name == f"take_{spec['key_b']}" and s["at"] == rooms[-2] \
                and not s[spec["key_b_state"]]:
            s[spec["key_b_state"]], ok, info = True, True, f"took the {spec['key_b']}"
        elif name == "unlock_a" and s["at"] == rooms[1] and s[spec["key_a_state"]]:
            s["door_a_open"], ok, info = True, True, "unlocked the first door"
        elif name == "unlock_b" and s["at"] == rooms[-2] and s[spec["key_b_state"]]:
            s["door_b_open"], ok, info = True, True, "unlocked the second door"
        elif name == spec["decoy"] and s["at"] == rooms[-2]:
            info = f"the {spec['decoy']} echoes, but the door does not move"
        elif name == f"take_{spec['treasure']}" and s["at"] == rooms[-1]:
            s[spec["treasure_state"]], ok, info = True, True, f"took the {spec['treasure']}"
        self.log.append(f"{name}{args or ''}: {info}")
        return ok, info


def _hard_treasure_actions(spec: dict) -> list[Action]:
    rooms = spec["rooms"]
    actions = []
    for a, b in zip(rooms, rooms[1:]):
        if b == rooms[-1]:
            pre = {"at": a, "door_b_open": True}
        elif b == rooms[-2]:
            pre = {"at": a, "door_a_open": True}
        else:
            pre = {"at": a}
        actions.append(Action(f"go_{b}_from_{a}", pre, {"at": b},
                              args={"_do": "go", "room": b}))
        actions.append(Action(f"go_{a}_from_{b}", {"at": b}, {"at": a},
                              args={"_do": "go", "room": a}))
    actions.append(Action(f"take_{spec['key_a']}",
                          {"at": rooms[1], spec["key_a_state"]: False},
                          {spec["key_a_state"]: True},
                          args={"_do": f"take_{spec['key_a']}"}))
    actions.append(Action("unlock_a",
                          {"at": rooms[1], spec["key_a_state"]: True},
                          {"door_a_open": True}, args={"_do": "unlock_a"}))
    actions.append(Action(f"take_{spec['key_b']}",
                          {"at": rooms[-2], spec["key_b_state"]: False},
                          {spec["key_b_state"]: True},
                          args={"_do": f"take_{spec['key_b']}"}))
    actions.append(Action("unlock_b",
                          {"at": rooms[-2], spec["key_b_state"]: True},
                          {"door_b_open": True}, args={"_do": "unlock_b"}))
    # The decoy: cheapest way to open door B on paper, never works.
    actions.append(Action(spec["decoy"], {"at": rooms[-2]},
                          {"door_b_open": True}, cost=0.3,
                          args={"_do": spec["decoy"]}))
    actions.append(Action(f"take_{spec['treasure']}",
                          {"at": rooms[-1]},
                          {spec["treasure_state"]: True},
                          args={"_do": f"take_{spec['treasure']}"}))
    return actions


class HardWorkshopWorld(Environment):
    """Hard workshop: two cuts, two tools, two moulds. The durable
    tool cuts both materials; the fragile tool cuts the soft material
    once and then breaks (P-B signal) and cannot cut the hard
    material at all. The press has three variants: a cheap mould-less
    one (fails), a decoy one whose mould key IS in the world's
    vocabulary (the decoy mould is a real takeable object in another
    room) but belongs to the wrong variant (fails), and the correct
    mould variant (works). ``spec`` carries rooms, tools, moulds,
    materials and the treasure."""

    def __init__(self, spec: dict):
        self.spec = spec
        rooms = spec["rooms"]
        self.adj = {r: [] for r in rooms}
        for a, b in zip(rooms, rooms[1:]):
            self.adj[a].append(b)
            self.adj[b].append(a)
        self.s = {"at": rooms[0], spec["durable_key"]: False,
                  spec["fragile_key"]: False, spec["mould_key"]: False,
                  spec["decoy_mould_key"]: False, "door_open": False,
                  "pressed": False, spec["treasure_key"]: False,
                  "fragile_broken": False}
        self.s.update({ck: False for _m, ck in spec["cuts"]})
        self.log: list[str] = []

    def observe(self) -> dict:
        return dict(self.s)

    def act(self, name: str, args: dict) -> tuple[bool, str]:
        spec, s = self.spec, self.s
        rooms = spec["rooms"]
        ok, info = False, "nothing happened"
        if name == "go":
            dest = args.get("room")
            if dest in self.adj[s["at"]] and (
                    dest != rooms[-1] or s["door_open"]):
                s["at"], ok, info = dest, True, f"moved to {dest}"
            else:
                info = f"cannot reach {dest}"
        elif name == f"take_{spec['durable']}" and s["at"] == rooms[1] \
                and not s[spec["durable_key"]]:
            s[spec["durable_key"]], ok, info = True, True, f"took the {spec['durable']}"
        elif name == f"take_{spec['fragile']}" and s["at"] == rooms[1] \
                and not s[spec["fragile_key"]]:
            s[spec["fragile_key"]], ok, info = True, True, f"took the {spec['fragile']}"
        elif name == f"take_{spec['mould']}" and s["at"] == rooms[-1] \
                and not s[spec["mould_key"]]:
            s[spec["mould_key"]], ok, info = True, True, f"took the {spec['mould']}"
        elif name == f"take_{spec['decoy_mould']}" and s["at"] == rooms[-2] \
                and not s[spec["decoy_mould_key"]]:
            s[spec["decoy_mould_key"]], ok, info = True, True, f"took the {spec['decoy_mould']}"
        elif name == "cut":
            material = args.get("material")
            cut_key = next((ck for m, ck in spec["cuts"] if m == material),
                           None)
            if cut_key is None or s[cut_key]:
                info = "nothing to cut here"
            elif material == spec["soft_material"] and s[spec["fragile_key"]] \
                    and not s["fragile_broken"]:
                s[cut_key], s["fragile_broken"] = True, True
                ok, info = True, (f"cut the {material} with the "
                                   f"{spec['fragile']}; it breaks")
            elif s[spec["durable_key"]]:
                s[cut_key], ok, info = True, True, f"cut the {material}"
            elif material == spec["hard_material"] and s[spec["fragile_key"]]:
                info = f"the {spec['fragile']} cannot cut {material}"
            else:
                info = "nothing to cut here"
        elif name == "pry_gate" and s["at"] == rooms[-2] and s[spec["durable_key"]]:
            s["door_open"], ok, info = True, True, "pried the gate open"
        elif name == "press" and s[spec["mould_key"]]:
            s["pressed"], ok, info = True, True, "the press stamps the parts"
        elif name == "press":
            info = f"the press has no {spec['mould']} seated"
        elif name == "assemble" and s["at"] == rooms[-1] and s["pressed"] \
                and all(s[ck] for _m, ck in spec["cuts"]):
            s[spec["treasure_key"]], ok, info = True, True, f"assembled the {spec['treasure']}"
        elif name == "assemble":
            info = "parts missing for assembly"
        self.log.append(f"{name}{args or ''}: {info}")
        return ok, info


def _hard_workshop_actions(spec: dict) -> list[Action]:
    rooms = spec["rooms"]
    dk, fk = spec["durable_key"], spec["fragile_key"]
    mk, dmk = spec["mould_key"], spec["decoy_mould_key"]
    actions = []
    for a, b in zip(rooms, rooms[1:]):
        gated = (b == rooms[-1])
        pre = {"at": a, "door_open": True} if gated else {"at": a}
        actions.append(Action(f"go_{b}_from_{a}", pre, {"at": b},
                              args={"_do": "go", "room": b}))
        actions.append(Action(f"go_{a}_from_{b}", {"at": b}, {"at": a},
                              args={"_do": "go", "room": a}))
    actions.append(Action(f"take_{spec['durable']}", {"at": rooms[1], dk: False},
                          {dk: True}, args={"_do": f"take_{spec['durable']}"}))
    actions.append(Action(f"take_{spec['fragile']}", {"at": rooms[1], fk: False},
                          {fk: True}, args={"_do": f"take_{spec['fragile']}"}))
    actions.append(Action(f"take_{spec['mould']}", {"at": rooms[-1], mk: False},
                          {mk: True}, args={"_do": f"take_{spec['mould']}"}))
    actions.append(Action(f"take_{spec['decoy_mould']}",
                          {"at": rooms[-2], dmk: False},
                          {dmk: True}, args={"_do": f"take_{spec['decoy_mould']}"}))
    for material, cut_key in spec["cuts"]:
        room = rooms[1] if material == spec["soft_material"] else rooms[-2]
        if material == spec["soft_material"]:
            # Cheap fragile first (works once, then breaks), durable
            # as the reliable alternative.
            actions.append(Action("cut", {"at": room, fk: True},
                                  {cut_key: True}, cost=0.5,
                                  args={"material": material}))
        else:
            # Cheap fragile first (cannot cut the hard material at
            # all), durable as the reliable alternative.
            actions.append(Action("cut", {"at": room, fk: True},
                                  {cut_key: True}, cost=0.5,
                                  args={"material": material}))
        actions.append(Action("cut", {"at": room, dk: True},
                              {cut_key: True}, cost=1.0,
                              args={"material": material}))
    actions.append(Action("pry_gate", {"at": rooms[-2], dk: True},
                          {"door_open": True}, cost=5.0,
                          args={"_do": "pry_gate"}))
    actions.append(Action("press", {"at": rooms[1]}, {"pressed": True},
                          cost=0.3, args={"_do": "press"}))
    actions.append(Action("press", {"at": rooms[-1], dmk: True},
                          {"pressed": True}, cost=1.0, args={"_do": "press"}))
    actions.append(Action("press", {"at": rooms[-1], mk: True},
                          {"pressed": True}, cost=2.0, args={"_do": "press"}))
    actions.append(Action(
        "assemble",
        {"at": rooms[-1], **{ck: True for _m, ck in spec["cuts"]},
         "pressed": True},
        {spec["treasure_key"]: True}, args={"_do": "assemble"}))
    return actions


class HardSurpriseWorld(Environment):
    """Hard surprise: two-stage warming. The coil goes cold -> warm
    -> hot; ``heat`` succeeds only when the coil is hot. The believed
    model knows ``warm_coil`` and ``stoke_coil`` as actions, but the
    naive plan is heat-then-warm-then-heat, so a fresh brain must fail
    at least once (heat on a merely warm coil) before it discovers
    the second stage."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.s = {"at": spec["rooms"][0], "coil": "cold",
                  spec["hot_key"]: False}
        self.log: list[str] = []

    def observe(self) -> dict:
        return dict(self.s)

    def act(self, name: str, args: dict) -> tuple[bool, str]:
        s = self.s
        ok, info = False, "nothing happened"
        if name == "warm_coil" and s["coil"] == "cold":
            s["coil"], ok, info = "warm", True, "the coil is warm now"
        elif name == "stoke_coil" and s["coil"] == "warm":
            s["coil"], ok, info = "hot", True, "the coil is hot now"
        elif name == "stoke_coil":
            info = "the coil is too cold to stoke"
        elif name == f"heat_{self.spec['substance']}" and s["coil"] == "hot":
            s[self.spec["hot_key"]], ok, info = True, True, (
                f"the {self.spec['substance']} is hot now")
        elif name == f"heat_{self.spec['substance']}":
            info = "the coil hums but the substance stays cold"
        self.log.append(f"{name}: {info}")
        return ok, info


def _hard_surprise_actions(spec: dict) -> list[Action]:
    return [
        Action(f"heat_{spec['substance']}", {"at": spec["rooms"][0]},
               {spec["hot_key"]: True}, cost=1.0,
               args={"_do": f"heat_{spec['substance']}"}),
        Action("warm_coil", {"at": spec["rooms"][0], "coil": "cold"},
               {"coil": "warm"}, cost=1.0, args={"_do": "warm_coil"}),
        Action("stoke_coil", {"at": spec["rooms"][0], "coil": "warm"},
               {"coil": "hot"}, cost=1.0, args={"_do": "stoke_coil"}),
    ]


def _generate_hard(seed: int, family: str, knobs: dict):
    """Hard-mode builder for ``generate`` (see its docstring). The
    rng is the same seeded stream the baseline builder uses, so a
    hard world is deterministic per (seed, family, knobs)."""
    rng = random.Random(seed)
    if family == "treasure":
        n_rooms = int(knobs.get("n_rooms") or rng.choice([4, 5]))
        rooms = rng.sample(_ROOMS, n_rooms)
        key_a, key_b = rng.sample(_KEYS, 2)
        treasure = rng.choice([t for t in _TREASURES
                               if t not in (key_a, key_b)])
        decoy = rng.choice(_DECOYS)
        spec = {"rooms": rooms, "key_a": key_a, "key_a_state": f"has_{key_a}",
                "key_b": key_b, "key_b_state": f"has_{key_b}",
                "treasure": treasure, "treasure_state": f"has_{treasure}",
                "decoy": decoy}
        env = HardTreasureWorld(spec)
        actions = _hard_treasure_actions(spec)
        goal = {spec["treasure_state"]: True}
        shape = {"family": family, "seed": seed, "rooms": list(rooms),
                 "min_steps": (n_rooms - 1) + 5,
                 "key_actions": [f"take_{key_a}", "unlock_a",
                                 f"take_{key_b}", "unlock_b",
                                 f"take_{treasure}"],
                 "order_constraints": [[f"take_{key_a}", "unlock_a"],
                                       ["unlock_a", f"take_{key_b}"],
                                       [f"take_{key_b}", "unlock_b"],
                                       ["unlock_b", f"take_{treasure}"]],
                 "n_objects": 3,
                 "difficulty": dict(knobs),
                 "n_decoys": 1 if knobs.get("decoy") else 0,
                 "n_gates": 2 if knobs.get("second_gate") else 1}
        return env, actions, goal, shape
    if family == "workshop":
        n_rooms = rng.choice([3, 4])
        rooms = rng.sample(_ROOMS, n_rooms)
        soft, hard = rng.sample(_MATERIALS, 2)
        cuts = [(soft, "cut_a"), (hard, "cut_b")]
        durable, fragile = rng.sample(_TOOLS, 2)
        mould, decoy_mould = rng.sample(_MOULDS, 2)
        treasure = rng.choice([t for t in _TREASURES if t != durable])
        spec = {"rooms": rooms, "durable": durable,
                "durable_key": f"has_{durable}", "fragile": fragile,
                "fragile_key": f"has_{fragile}", "mould": mould,
                "mould_key": f"has_{mould}", "decoy_mould": decoy_mould,
                "decoy_mould_key": f"has_{decoy_mould}", "cuts": cuts,
                "soft_material": soft, "hard_material": hard,
                "treasure": treasure, "treasure_key": f"has_{treasure}"}
        env = HardWorkshopWorld(spec)
        actions = _hard_workshop_actions(spec)
        goal = {spec["treasure_key"]: True}
        shape = {"family": family, "seed": seed, "rooms": list(rooms),
                 # go x(n-1) + take durable + take mould + 2 cuts
                 # + pry + press + assemble (durable-only minimum;
                 # the fragile route is an alternative, not required).
                 "min_steps": (n_rooms - 1) + 7,
                 "key_actions": [f"take_{durable}", f"take_{mould}",
                                 "cut", "pry_gate", "press", "assemble"],
                 "order_constraints": [[f"take_{durable}", "cut"],
                                       ["pry_gate", f"take_{mould}"],
                                       [f"take_{mould}", "press"],
                                       ["press", "assemble"],
                                       ["cut", "assemble"]],
                 "n_objects": 4,
                 "difficulty": dict(knobs),
                 "n_decoys": 1 if knobs.get("decoy_variant") else 0,
                 "breakable_tool": bool(knobs.get("breakable_tool"))}
        return env, actions, goal, shape
    # surprise
    room = rng.choice(_ROOMS)
    substance = rng.choice(_SUBSTANCES)
    spec = {"rooms": [room], "substance": substance,
            "hot_key": f"{substance}_hot"}
    env = HardSurpriseWorld(spec)
    actions = _hard_surprise_actions(spec)
    goal = {spec["hot_key"]: True}
    shape = {"family": family, "seed": seed, "rooms": [room],
             "min_steps": 3,
             "key_actions": ["warm_coil", "stoke_coil", f"heat_{substance}"],
             "order_constraints": [["warm_coil", "stoke_coil"],
                                   ["stoke_coil", f"heat_{substance}"]],
             "n_objects": 1,
             "difficulty": dict(knobs),
             "n_hidden_preconditions": 2 if knobs.get("two_stage") else 1}
    return env, actions, goal, shape


def generate_hard(seed: int, family: str, difficulty="hard"):
    """Build a hard generated environment: ``generate(seed, family,
    difficulty)`` with the family's full hard knobs by default.
    Same return shape as :func:`generate`."""
    if family not in FAMILIES:
        raise ValueError(f"unknown envgen family: {family!r} "
                         f"(known: {sorted(FAMILIES)})")
    return _generate_hard(seed, family,
                          _resolve_difficulty(family, difficulty))


# ------------------------------------------------------------------ generator
def generate(seed: int, family: str, difficulty=None):
    """Build a fresh environment of ``family`` from ``seed``.

    Returns ``(env, actions, goal, expected_solution_shape)`` where
    ``goal`` is the target state dict and the shape is
    ``{"family", "seed", "rooms", "min_steps", "key_actions",
    "order_constraints", "n_objects"}``: the minimum step count of
    the constructed solution, the actions it must include, and the
    (before, after) pairs its order must respect. Raises
    ``ValueError`` for an unknown family.

    ``difficulty`` (Phase 10, P10-2): ``None`` (default) keeps the
    exact baseline behaviour; ``"hard"`` or a knob dict (see
    :func:`_resolve_difficulty`) builds the harder world for the
    family -- two gates + decoy opener (treasure), decoy press
    variant + breakable tool (workshop), two-stage warming
    (surprise) -- and the shape gains a ``"difficulty"`` entry
    recording the knobs that were used."""
    if family not in FAMILIES:
        raise ValueError(f"unknown envgen family: {family!r} "
                         f"(known: {sorted(FAMILIES)})")
    if difficulty is not None:
        return _generate_hard(seed, family,
                              _resolve_difficulty(family, difficulty))
    rng = random.Random(seed)
    if family == "treasure":
        n_rooms = rng.choice([3, 3, 4])
        rooms = rng.sample(_ROOMS, n_rooms)
        key = rng.choice(_KEYS)
        treasure = rng.choice([t for t in _TREASURES if t != key])
        spec = {"rooms": rooms, "key": key,
                "key_state": f"has_{key}", "treasure": treasure,
                "treasure_state": f"has_{treasure}"}
        env = GeneratedTreasureWorld(spec)
        actions = _treasure_actions(spec)
        goal = {spec["treasure_state"]: True}
        # go x(n-1) + take key + unlock + take treasure.
        shape = {"family": family, "seed": seed, "rooms": list(rooms),
                 "min_steps": (n_rooms - 1) + 3,
                 "key_actions": [f"take_{key}", "unlock",
                                 f"take_{treasure}"],
                 "order_constraints": [[f"take_{key}", "unlock"],
                                       ["unlock", f"take_{treasure}"]],
                 "n_objects": 2}
        return env, actions, goal, shape
    if family == "workshop":
        n_rooms = rng.choice([3, 4])
        rooms = rng.sample(_ROOMS, n_rooms)
        n_cuts = n_rooms - 1
        materials = rng.sample(_MATERIALS, n_cuts)
        cuts = [(m, f"cut_{chr(ord('a') + i)}")
                for i, m in enumerate(materials)]
        tool = rng.choice(_TOOLS)
        mould = rng.choice(_MOULDS)
        treasure = rng.choice([t for t in _TREASURES if t != tool])
        spec = {"rooms": rooms, "tool": tool, "tool_key": f"has_{tool}",
                "mould": mould, "mould_key": f"has_{mould}",
                "cuts": cuts, "treasure": treasure,
                "treasure_key": f"has_{treasure}"}
        env = GeneratedWorkshopWorld(spec)
        actions = _workshop_actions(spec)
        goal = {spec["treasure_key"]: True}
        # go x(n-1) + take tool + take mould + n cuts + pry + press
        # + assemble (the gate must be opened before the last room,
        # the mould fetched before the press).
        shape = {"family": family, "seed": seed, "rooms": list(rooms),
                 "min_steps": (n_rooms - 1) + 2 + n_cuts + 3,
                 "key_actions": [f"take_{tool}", f"take_{mould}",
                                 "cut", "pry_gate", "press", "assemble"],
                 "order_constraints": [[f"take_{tool}", "cut"],
                                       ["pry_gate", f"take_{mould}"],
                                       [f"take_{mould}", "press"],
                                       ["press", "assemble"],
                                       ["cut", "assemble"]],
                 "n_objects": 2 + n_cuts}
        return env, actions, goal, shape
    # surprise
    room = rng.choice(_ROOMS)
    substance = rng.choice(_SUBSTANCES)
    spec = {"rooms": [room], "substance": substance,
            "hot_key": f"{substance}_hot"}
    env = GeneratedSurpriseWorld(spec)
    actions = _surprise_actions(spec)
    goal = {spec["hot_key"]: True}
    shape = {"family": family, "seed": seed, "rooms": [room],
             "min_steps": 2,
             "key_actions": ["warm_coil", f"heat_{substance}"],
             "order_constraints": [["warm_coil", f"heat_{substance}"]],
             "n_objects": 1}
    return env, actions, goal, shape
