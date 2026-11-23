"""Environments the brain can sense and act on, plus a small toy world for testing."""
from __future__ import annotations

from ..planning.planner import Action


class Environment:
    def observe(self) -> dict:
        raise NotImplementedError

    def act(self, name: str, args: dict) -> tuple[bool, str]:
        raise NotImplementedError


class RoomWorld(Environment):
    """hall - lab - vault. The vault door (in the lab) needs a key found in the lab.
    'pick_lock' looks easier on paper but never works in reality: the brain must learn that."""
    ADJ = {"hall": ["lab"], "lab": ["hall", "vault"], "vault": ["lab"]}

    def __init__(self):
        self.s = {"at": "hall", "has_key": False, "door_open": False, "has_treasure": False}
        self.log: list[str] = []

    def observe(self) -> dict:
        return dict(self.s)

    def act(self, name: str, args: dict) -> tuple[bool, str]:
        s = self.s
        ok, info = False, "nothing happened"
        if name == "go":
            dest = args.get("room")
            if dest in self.ADJ[s["at"]] and (dest != "vault" or s["door_open"]):
                s["at"], ok, info = dest, True, f"moved to {dest}"
            else:
                info = f"cannot reach {dest}"
        elif name == "take_key" and s["at"] == "lab" and not s["has_key"]:
            s["has_key"], ok, info = True, True, "took the key"
        elif name == "unlock" and s["at"] == "lab" and s["has_key"]:
            s["door_open"], ok, info = True, True, "unlocked the vault door"
        elif name == "pick_lock":
            info = "the lock resists; picking fails"
        elif name == "take_treasure" and s["at"] == "vault":
            s["has_treasure"], ok, info = True, True, "took the treasure"
        self.log.append(f"{name}{args or ''}: {info}")
        return ok, info


def room_world_actions() -> list[Action]:
    """The brain's (initially naive) belief about how this world works."""
    acts = [
        Action("go_lab_from_hall", {"at": "hall"}, {"at": "lab"}, args={"_do": "go", "room": "lab"}),
        Action("go_hall_from_lab", {"at": "lab"}, {"at": "hall"}, args={"_do": "go", "room": "hall"}),
        Action("go_vault", {"at": "lab", "door_open": True}, {"at": "vault"}, args={"_do": "go", "room": "vault"}),
        Action("take_key", {"at": "lab", "has_key": False}, {"has_key": True}, args={"_do": "take_key"}),
        Action("unlock", {"at": "lab", "has_key": True}, {"door_open": True}, args={"_do": "unlock"}),
        Action("pick_lock", {"at": "lab"}, {"door_open": True}, args={"_do": "pick_lock"}),
        Action("take_treasure", {"at": "vault"}, {"has_treasure": True}, args={"_do": "take_treasure"}),
    ]
    return acts
