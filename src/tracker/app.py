"""
Riichi Mahjong State Tracker (Mahjong Soul rules)
Two-page Tkinter UI (Setup → Game) with button-only inputs.

Page 1 — Setup:
  • Choose our seat (East/South/West/North)
  • Choose initial dora indicator (one tile for now)
  • Choose starting hand (by pressing tile buttons; shows count)
  • Start Game → initializes state and moves to Page 2

Page 2 — Game:
  • Choose INITIATOR (E/S/W/N)
  • Choose OPERATION type (Discard, Draw, Chi, Pon, Kan, Added Kan, Concealed Kan, Riichi, Ron, Tsumo, Dora Flip, Pass)
  • Choose involved tiles via buttons (tiles panel + quick-clear)
  • Apply Operation → updates tracker, advances turn with simple logic
  • If it becomes OUR turn (or we can call on a discard), we show a placeholder: "AI decision point"

Notes
-----
* Mahjong Soul specifics like abortive draws, exhaustive draw, furiten legality, chi restriction, added-kan rob, etc., are not yet enforced. This is a lightweight driver to hook your AI.
* Tile notation: "1-9m/p/s", "0m/p/s" = red fives, honors "1z..7z" (E,S,W,N, white, green, red).

Run
---
python app.py

Dependencies: Python 3.9+ standard library.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

Tile = str  # e.g., "5m", "0p", "1z".."7z"
SEAT_NAMES = ["E", "S", "W", "N"]

# ---------------------------
# Utilities
# ---------------------------

def normalize_tile(t: str) -> Tile:
    t = t.strip().lower()
    if len(t) < 2:
        raise ValueError(f"Bad tile: {t}")
    suit = t[-1]
    rank = t[:-1]
    if suit not in "mpsz":
        raise ValueError(f"Bad suit: {t}")
    if suit == "z":
        if rank not in {"1","2","3","4","5","6","7"}:
            raise ValueError(f"Bad honor: {t}")
    else:
        if rank not in {"0","1","2","3","4","5","6","7","8","9"}:
            raise ValueError(f"Bad suit rank: {t}")
    return f"{rank}{suit}"


def tile_list_str(lst: List[Tile]) -> str:
    return " ".join(lst) if lst else "(none)"

# simple suit helpers
ALL_TILES = [
    *[f"{i}m" for i in [0,1,2,3,4,5,6,7,8,9]],
    *[f"{i}p" for i in [0,1,2,3,4,5,6,7,8,9]],
    *[f"{i}s" for i in [0,1,2,3,4,5,6,7,8,9]],
    *[f"{i}z" for i in [1,2,3,4,5,6,7]],
]
SUIT_GROUPS: List[Tuple[str, List[Tile]]] = [
    ("Manzu", [f"{i}m" for i in [1,2,3,4,5,6,7,8,9]]),
    ("Pinzu", [f"{i}p" for i in [1,2,3,4,5,6,7,8,9]]),
    ("Souzu", [f"{i}s" for i in [1,2,3,4,5,6,7,8,9]]),
    ("Reds", ["0m","0p","0s"]),
    ("Honors", [f"{i}z" for i in [1,2,3,4,5,6,7]]),
]

# ---------------------------
# Domain model
# ---------------------------

@dataclass
class Meld:
    kind: str  # CHI | PON | KAN | ADDED_KAN | CONCEALED_KAN
    tiles: List[Tile]
    from_who: Optional[int] = None

@dataclass
class Player:
    seat: int
    wind: str  # E/S/W/N
    riichi: bool = False
    open_melds: List[Meld] = field(default_factory=list)
    discards: List[Tile] = field(default_factory=list)
    concealed: List[Tile] = field(default_factory=list)  # only tracked for our seat

@dataclass
class Wall:
    tiles_left: int = 70
    kan_count: int = 0
    dora_indicators: List[Tile] = field(default_factory=list)

@dataclass
class RoundInfo:
    kyoku: int = 1
    honba: int = 0
    wind_round: str = "E"
    dealer: int = 0
    riichi_sticks: int = 0

@dataclass
class TableState:
    round: RoundInfo = field(default_factory=RoundInfo)
    wall: Wall = field(default_factory=Wall)
    players: List[Player] = field(default_factory=list)
    our_seat: int = 0
    current_turn: int = 0  # whose action to draw/discard next

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

# ---------------------------
# Tracker with minimal turn logic
# ---------------------------

class StateTracker:
    def __init__(self):
        self.state = TableState(players=[Player(seat=i, wind=SEAT_NAMES[i]) for i in range(4)])
        self.last_discard: Optional[Tuple[int, Tile]] = None  # (actor, tile)

    def next_seat(self, s: int) -> int:
        return (s + 1) % 4

    # ---------- legality helpers ----------
    def _ensure_turn(self, actor: int, op: str):
        if actor != self.state.current_turn:
            raise ValueError(f"Illegal {op}: it's {SEAT_NAMES[self.state.current_turn]}'s turn, not {SEAT_NAMES[actor]}")

    def _ensure_has_tile(self, actor: int, tile: Tile):
        t = normalize_tile(tile)
        if actor == self.state.our_seat:
            hand = self.state.players[self.state.our_seat].concealed
            if hand.count(t) == 0:
                raise ValueError(f"Illegal discard: our hand does not contain {t}")
        # For opponents we don't track concealed; allow

    # ---------- ops ----------
    def apply_draw(self, actor: int, tile: Optional[Tile] = None):
        self._ensure_turn(actor, "DRAW")
        if self.state.wall.tiles_left > 0:
            self.state.wall.tiles_left -= 1
        if actor == self.state.our_seat and tile:
            self.state.players[self.state.our_seat].concealed.append(normalize_tile(tile))
        # After a draw, same actor to discard
        self.state.current_turn = actor
        self.last_discard = None

    def apply_discard(self, actor: int, tile: Tile):
        self._ensure_turn(actor, "DISCARD")
        t = normalize_tile(tile)
        self._ensure_has_tile(actor, t)
        p = self.state.players[actor]
        p.discards.append(t)
        if actor == self.state.our_seat:
            if t in p.concealed:
                p.concealed.remove(t)
        # After a discard, opportunities for calls may arise. Turn advances if no one calls.
        self.state.current_turn = self.next_seat(actor)
        self.last_discard = (actor, t)

    def apply_call(self, actor: int, kind: str, tiles: List[Tile], from_who: Optional[int]):
        # Calls must immediately follow a discard
        if not self.last_discard:
            raise ValueError("Illegal call: no prior discard to call on")
        discarder, last_tile = self.last_discard
        # Chi restriction: only left player of discarder can chi
        if kind == "CHI" and self.next_seat(discarder) != actor:
            raise ValueError("Illegal chi: only the player to discarder’s left may chi")
        norm_tiles = [normalize_tile(x) for x in tiles]
        if actor == self.state.our_seat:
            hand = self.state.players[self.state.our_seat].concealed
            if kind == "CHI":
                need = [t for t in norm_tiles if t != normalize_tile(last_tile)]
                if len(need) != 2 or any(hand.count(t) < need.count(t) for t in set(need)):
                    raise ValueError("Illegal chi: insufficient tiles in hand")
                for t in need:
                    hand.remove(t)
            elif kind == "PON":
                if hand.count(normalize_tile(last_tile)) < 2:
                    raise ValueError("Illegal pon: need two matching tiles in hand")
                # remove two
                for _ in range(2):
                    hand.remove(normalize_tile(last_tile))
            elif kind in {"KAN", "ADDED_KAN"}:
                if hand.count(normalize_tile(last_tile)) < 3:
                    raise ValueError("Illegal kan: need three in hand matching the discard")
                for _ in range(3):
                    hand.remove(normalize_tile(last_tile))
            elif kind == "CONCEALED_KAN":
                # simple check: four of a kind in hand
                if hand.count(norm_tiles[0]) < 4:
                    raise ValueError("Illegal concealed kan: need four identical tiles in hand")
                for _ in range(4):
                    hand.remove(norm_tiles[0])
        meld = Meld(kind=kind, tiles=norm_tiles, from_who=discarder)
        self.state.players[actor].open_melds.append(meld)
        if kind in {"KAN", "ADDED_KAN", "CONCEALED_KAN"}:
            self.state.wall.kan_count += 1
        # Caller becomes next to discard
        self.state.current_turn = actor
        # Last discard consumed by the call
        self.last_discard = None

    def apply_riichi(self, actor: int):
        self._ensure_turn(actor, "RIICHI")
        self.state.players[actor].riichi = True
        self.state.round.riichi_sticks += 1
        self.state.current_turn = actor

    def apply_dora_flip(self, indicator: Tile):
        self.state.wall.dora_indicators.append(normalize_tile(indicator))

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Riichi Tracker (Mahjong Soul) — v0.3")
        self.geometry("1100x760")
        self.tracker = StateTracker()
        self.selected_tiles: List[Tile] = []  # used on both pages for inputs
        self.tile_buttons: Dict[str, ttk.Button] = {}  # map tile->button on Game page
        self._build_pages()
        self.show_frame("SetupPage")

    # ---- Navigation ----
    def _build_pages(self):
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True)
        self.frames: Dict[str, tk.Frame] = {}
        for F in (SetupPage, GamePage):
            frame = F(parent=container, controller=self)
            self.frames[F.__name__] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

    def show_frame(self, name: str):
        frame = self.frames[name]
        if hasattr(frame, 'on_show'):
            frame.on_show()
        frame.tkraise()

    # ---- Tile button panel ----
    def make_tile_panel(self, parent: tk.Widget, on_click, register_buttons: bool=False):
        outer = ttk.Frame(parent)
        for gi, (label, tiles) in enumerate(SUIT_GROUPS):
            lf = ttk.LabelFrame(outer, text=label)
            lf.grid(row=gi, column=0, sticky="w", padx=4, pady=4)
            for col, t in enumerate(tiles):
                b = ttk.Button(lf, text=t, width=4, command=lambda tt=t: on_click(tt))
                b.grid(row=0, column=col, padx=2, pady=2)
                if register_buttons:
                    self.tile_buttons[t] = b
        return outer

    def reset_selection(self):
        self.selected_tiles = []

    def add_selection(self, t: Tile):
        self.selected_tiles.append(normalize_tile(t))

    def update_tile_highlight(self):
        """On our turn: enable buttons that are in our hand; disable others. Otherwise enable all."""
        st = self.tracker.state
        if not getattr(self, 'tile_buttons', None):
            return
        if st.current_turn == st.our_seat:
            hand = st.players[st.our_seat].concealed
            hand_set = set(hand)
            for tile, btn in self.tile_buttons.items():
                if tile in hand_set:
                    btn.state(["!disabled"])  # enabled
                else:
                    btn.state(["disabled"])  # dim
        else:
            for btn in self.tile_buttons.values():
                btn.state(["!disabled"])  # enable all when not our turn

# ---------------------------
# Setup Page
# ---------------------------

class SetupPage(ttk.Frame):
    def __init__(self, parent, controller: App):
        super().__init__(parent)
        self.controller = controller
        self.our_seat_var = tk.IntVar(value=0)
        self.dora_var: Optional[Tile] = None
        self.status = tk.StringVar(value="Pick seat, dora, and 13 tiles (or 14 if dealer).")
        self._build()

    def _build(self):
        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.X)

        # Seat selection
        ttk.Label(top, text="Our Seat:").pack(side=tk.LEFT)
        for i, name in enumerate(SEAT_NAMES):
            ttk.Radiobutton(top, text=name, value=i, variable=self.our_seat_var).pack(side=tk.LEFT, padx=4)

        # Dora indicator (single for start)
        self.dora_label = ttk.Label(top, text="Dora: (none)")
        self.dora_label.pack(side=tk.LEFT, padx=16)
        ttk.Button(top, text="Clear Dora", command=self.clear_dora).pack(side=tk.LEFT)

        # Hand summary
        self.hand_label = ttk.Label(top, text="Hand: (empty)")
        self.hand_label.pack(side=tk.RIGHT)

        mid = ttk.Frame(self)
        mid.pack(fill=tk.BOTH, expand=True)

        # Left: tile panel
        tile_panel = self.controller.make_tile_panel(mid, self.on_tile_click, register_buttons=True)
        tile_panel.pack(side=tk.LEFT, anchor="n", padx=8, pady=8)

        # Right: controls
        right = ttk.Frame(mid, padding=8)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(right, textvariable=self.status, foreground="#444").pack(anchor="w")
        ttk.Separator(right).pack(fill=tk.X, pady=6)

        ttk.Button(right, text="Set as Dora", command=self.set_dora_from_selection).pack(anchor="w", pady=2)
        ttk.Button(right, text="Add to Hand", command=self.add_to_hand_from_selection).pack(anchor="w", pady=2)
        ttk.Button(right, text="Remove Last", command=self.remove_last_from_hand).pack(anchor="w", pady=2)
        ttk.Button(right, text="Clear Hand", command=self.clear_hand).pack(anchor="w", pady=2)
        ttk.Button(right, text="Start Game ▶", command=self.start_game).pack(anchor="w", pady=(12,2))

        self.selection_label = ttk.Label(right, text="Selection: (none)")
        self.selection_label.pack(anchor="w", pady=(12,0))

    # ---- Hand management ----
    def on_tile_click(self, t: Tile):
        self.controller.add_selection(t)
        self.selection_label.config(text=f"Selection: {tile_list_str(self.controller.selected_tiles)}")

    def set_dora_from_selection(self):
        if not self.controller.selected_tiles:
            messagebox.showinfo("Dora", "Click a tile, then 'Set as Dora'.")
            return
        self.dora_var = self.controller.selected_tiles[-1]
        self.controller.reset_selection()
        self.dora_label.config(text=f"Dora: {self.dora_var}")

    def add_to_hand_from_selection(self):
        if not self.controller.selected_tiles:
            messagebox.showinfo("Hand", "Click tiles to add.")
            return
        p = self.controller.tracker.state.players[0]  # temp; we'll assign after start
        # store in temp on the page object first
        if not hasattr(self, 'temp_hand'):
            self.temp_hand: List[Tile] = []
        self.temp_hand.extend(self.controller.selected_tiles)
        self.controller.reset_selection()
        self._refresh_hand_label()

    def remove_last_from_hand(self):
        if hasattr(self, 'temp_hand') and self.temp_hand:
            self.temp_hand.pop()
            self._refresh_hand_label()

    def clear_hand(self):
        self.temp_hand = []
        self._refresh_hand_label()

    def _refresh_hand_label(self):
        n = len(getattr(self, 'temp_hand', []))
        self.hand_label.config(text=f"Hand ({n}): {tile_list_str(getattr(self, 'temp_hand', []))}")

    def clear_dora(self):
        self.dora_var = None
        self.dora_label.config(text="Dora: (none)")

    def start_game(self):
        # Validate
        our_seat = int(self.our_seat_var.get())
        dealer = 0  # we'll rotate winds so that seat indices still E,S,W,N order
        # Construct players with winds fixed E,S,W,N and set our seat
        self.controller.tracker.state = TableState(
            players=[Player(seat=i, wind=SEAT_NAMES[i]) for i in range(4)],
            our_seat=our_seat,
            current_turn=0,
        )
        st = self.controller.tracker.state
        st.round.dealer = 0  # simplify: seat 0 is East by convention; our seat may be != 0
        # Place starting hand into our seat's concealed tiles
        hand = getattr(self, 'temp_hand', [])
        if not hand:
            messagebox.showerror("Start", "Please add tiles to your hand.")
            return
        st.players[our_seat].concealed = [normalize_tile(x) for x in hand]
        # Dora
        if self.dora_var:
            st.wall.dora_indicators = [normalize_tile(self.dora_var)]
        # Wall tiles left — approximate start depending on dealer draw (not exact; okay for tracker)
        st.wall.tiles_left = 70
        # Show next page
        self.controller.show_frame("GamePage")

    def on_show(self):
        # Reset temp vars
        self.controller.reset_selection()
        self.temp_hand = []
        self._refresh_hand_label()
        self.clear_dora()
        self.status.set("Pick seat, dora, and 13 tiles (or 14 if dealer).")

# ---------------------------
# Game Page
# ---------------------------

class GamePage(ttk.Frame):
    OPS = ["DRAW","DISCARD","CHI","PON","KAN","ADDED_KAN","CONCEALED_KAN","RIICHI","RON","TSUMO","DORA_FLIP","PASS"]

    def __init__(self, parent, controller: App):
        super().__init__(parent)
        self.controller = controller
        self.actor_var = tk.IntVar(value=0)
        self.op_var = tk.StringVar(value="DISCARD")
        self.info = tk.StringVar(value="")
        self.last_discard: Optional[Tuple[int, Tile]] = None
        self._build()

    def save_json(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.controller.tracker.state.to_json())
            messagebox.showinfo("Saved", f"State saved to {path}")
        except Exception as e:
            messagebox.showerror("Save", str(e))

    def _build(self):
        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.X)

        ttk.Button(top, text="◀ Back to Setup", command=lambda: self.controller.show_frame("SetupPage")).pack(side=tk.LEFT)
        ttk.Label(top, textvariable=self.info).pack(side=tk.LEFT, padx=16)
        ttk.Button(top, text="Save State JSON", command=self.save_json).pack(side=tk.RIGHT)

        ctl = ttk.Frame(self, padding=8)
        ctl.pack(fill=tk.X)

        # Actor
        ttk.Label(ctl, text="Initiator:").grid(row=0, column=0, sticky="w")
        for i, name in enumerate(SEAT_NAMES):
            ttk.Radiobutton(ctl, text=name, value=i, variable=self.actor_var).grid(row=0, column=1+i)

        # Operation
        ttk.Label(ctl, text="Operation:").grid(row=1, column=0, sticky="w")
        op_box = ttk.Combobox(ctl, textvariable=self.op_var, values=self.OPS, width=18)
        op_box.grid(row=1, column=1, columnspan=2, sticky="w")

        # Selected tiles display
        self.sel_label = ttk.Label(ctl, text="Tiles: (none)")
        self.sel_label.grid(row=2, column=0, columnspan=3, sticky="w", pady=(6,0))
        ttk.Button(ctl, text="Clear Tiles", command=self.clear_tiles).grid(row=2, column=3, sticky="w")

        mid = ttk.Frame(self)
        mid.pack(fill=tk.BOTH, expand=True)

        tile_panel = self.controller.make_tile_panel(mid, self.on_tile_click)
        tile_panel.pack(side=tk.LEFT, anchor="n", padx=8, pady=8)

        right = ttk.Frame(mid, padding=8)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Button(right, text="Apply Operation", command=self.apply_operation, width=24).pack(anchor="w", pady=(4,8))
        self.state_text = tk.Text(right, height=28)
        self.state_text.pack(fill=tk.BOTH, expand=True)

    def on_show(self):
        self.controller.reset_selection()
        self.last_discard = None
        self.refresh_state_pane()
        self.update_info()

    # --- helpers ---
    def update_info(self):
        st = self.controller.tracker.state
        me = st.our_seat
        turn = st.current_turn
        msg = f"Our seat: {SEAT_NAMES[me]}  |  Current turn: {SEAT_NAMES[turn]}  |  Dora: {tile_list_str(st.wall.dora_indicators)}  |  Tiles left: {st.wall.tiles_left}"
        self.info.set(msg)
        self.controller.update_tile_highlight()

    def on_tile_click(self, t: Tile):
        self.controller.add_selection(t)
        self.sel_label.config(text=f"Tiles: {tile_list_str(self.controller.selected_tiles)}")

    def clear_tiles(self):
        self.controller.reset_selection()
        self.sel_label.config(text="Tiles: (none)")

    def apply_operation(self):
        st = self.controller.tracker.state
        tr = self.controller.tracker
        actor = int(self.actor_var.get())
        op = self.op_var.get().upper()
        tiles = [normalize_tile(x) for x in self.controller.selected_tiles]
        try:
            if op == "DRAW":
                tr.apply_draw(actor, tiles[0] if tiles else None)
                self.last_discard = None
            elif op == "DISCARD":
                if not tiles:
                    raise ValueError("DISCARD needs one tile")
                tr.apply_discard(actor, tiles[0])
                self.last_discard = tr.last_discard
                if self.last_discard and actor != st.our_seat:
                    self._maybe_call_prompt(actor, self.last_discard[1])
            elif op in {"CHI","PON","KAN","ADDED_KAN","CONCEALED_KAN"}:
                if not tiles:
                    raise ValueError(f"{op} needs tiles")
                tr.apply_call(actor, op, tiles, None)
                self.last_discard = None
            elif op == "RIICHI":
                tr.apply_riichi(actor)
            elif op == "DORA_FLIP":
                if not tiles:
                    raise ValueError("DORA_FLIP needs indicator tile")
                tr.apply_dora_flip(tiles[0])
            elif op in {"RON","TSUMO","PASS"}:
                pass
            else:
                raise ValueError(f"Unknown op {op}")
        except Exception as e:
            messagebox.showerror("Apply", str(e))
            return

        if st.current_turn == st.our_seat:
            self.ai_decision_point()
        self.controller.reset_selection()
        self.sel_label.config(text="Tiles: (none)")
        self.refresh_state_pane()
        self.update_info()

    def _maybe_call_prompt(self, discarder: int, tile: Tile):
        st = self.controller.tracker.state
        tr = self.controller.tracker
        can_calls = []
        if tr.can_pon(tile):
            can_calls.append("PON")
        if tr.can_chi(discarder, tile):
            can_calls.append("CHI")
        if tr.can_kan_from_discard(tile):
            can_calls.append("KAN")
        if can_calls:
            messagebox.showinfo("AI decision point", f"Opponent discarded {tile}. Possible calls for us: {', '.join(can_calls)}. (AI not implemented)")

    def ai_decision_point(self):
        st = self.controller.tracker.state
        me = st.our_seat
        hand = st.players[me].concealed
        # Placeholder – here you'd call your policy to choose: discard, riichi, kan, etc.
        messagebox.showinfo("AI decision point", f"It's our ({SEAT_NAMES[me]}) turn. Hand: {tile_list_str(hand)}. (AI not implemented)")

    def refresh_state_pane(self):
        st = self.controller.tracker.state
        txt = self.state_text
        txt.delete("1.0", tk.END)
        txt.insert(tk.END, f"Round: {st.round.wind_round}{st.round.kyoku}  Honba:{st.round.honba}  Dealer:{SEAT_NAMES[st.round.dealer]}  Riichi sticks:{st.round.riichi_sticks}")
        txt.insert(tk.END, f"Wall: tiles_left={st.wall.tiles_left} kan_count={st.wall.kan_count} dora={tile_list_str(st.wall.dora_indicators)}")
        for i, p in enumerate(st.players):
            txt.insert(tk.END, f"Player {SEAT_NAMES[i]} ({'US' if i==st.our_seat else 'OPP'}): riichi={p.riichi}")
            if i == st.our_seat:
                txt.insert(tk.END, f"  Concealed: {tile_list_str(p.concealed)}")
            txt.insert(tk.END, f"  Discards: {tile_list_str(p.discards)}")
            if p.open_melds:
                for m in p.open_melds:
                    txt.insert(tk.END, f"  Meld: {m.kind} from {m.from_who} :: {tile_list_str(m.tiles)}")
            txt.insert(tk.END, "")

# ---------------------------
# Entrypoint
# ---------------------------

if __name__ == "__main__":
    App().mainloop()
