"""
Backgammon with Expectiminimax AI - Graphical Interface (Step 3)
Name:       Krishna Mahajan
Student ID: 106wucww
Module:     5ENT1140 Artificial Intelligence Principles - Mini Project

Run with:  python3 main.py   (or python3 gui.py)
You play White (moving from point 24 down to 1). The AI plays Black.
Click one of your checkers, then click a highlighted destination.
Space = roll, H = hint (the Hard AI suggests your move), M = sound on/off.
The "AI thinking" box in the side panel shows the AI's top candidate turns and their values.
"""

import io
import sys
import math
import time
import random
import asyncio
import threading
from array import array
import pygame

from backgammon import (WHITE, BLACK, GameState, roll_dice, get_legal_turns, apply_move,
                        format_move)
from ai import easy_ai, medium_ai, hard_ai, pip_count, order_turns, chance_node, WIN_SCORE

# True when running in a web browser (pygbag / WebAssembly), where threads are not available
IN_BROWSER = sys.platform == "emscripten"

STUDENT_NAME = "Krishna Mahajan"
STUDENT_ID = "106wucww"

# ---------------------------------------------------------------------------
# Layout and colours
# ---------------------------------------------------------------------------
WIN_W, WIN_H = 1180, 720
BX, BY = 40, 70                       # top-left corner of the playing area
PW = 64                               # width of one point (triangle)
BAR_W = 56                            # width of the central bar
BOARD_W = 12 * PW + BAR_W
BOARD_H = 580
TRI_H = 240                           # triangle height
R = 26                                # checker radius
TRAY_X = BX + BOARD_W + 14            # bear-off tray
TRAY_W = 56
PANEL_X = TRAY_X + TRAY_W + 26        # side panel

FELT = (28, 70, 52)
FRAME = (92, 58, 30)
FRAME_DARK = (60, 36, 18)
POINT_A = (214, 180, 120)
POINT_B = (140, 44, 44)
BG = (22, 26, 32)
PANEL = (32, 38, 46)
TEXT = (235, 235, 235)
MUTED = (150, 158, 170)
GOLD = (255, 200, 60)
GREEN = (80, 220, 120)
CYAN = (90, 200, 255)
WHITE_CHK = (242, 238, 228)
BLACK_CHK = (34, 34, 38)

AI_LEVELS = {"Easy": easy_ai, "Medium": medium_ai, "Hard": hard_ai}
# (depth, beam) used by each level - must match medium_ai / hard_ai in ai.py
AI_SEARCH = {"Medium": (1, None), "Hard": (2, 6)}
HINT_SEARCH = AI_SEARCH["Hard"]
ROLL_ANIM_MS = 600
AI_MOVE_MS = 550
MOVE_ANIM_MS = 320                    # time for one checker to slide to its new point
HUMAN_ANIM_MS = 200


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def point_column(idx):
    """Returns the screen column (0-11, left to right) of point index 'idx'."""
    return 11 - idx if idx <= 11 else idx - 12


def point_center_x(idx):
    """Returns the x pixel of the centre of point 'idx' (the bar sits between columns 5 and 6)."""
    col = point_column(idx)
    return BX + col * PW + (BAR_W if col >= 6 else 0) + PW // 2


def checker_pos(idx, k):
    """Returns the (x, y) pixel centre of the k-th checker (0 = outermost) stacked on point 'idx'."""
    x = point_center_x(idx)
    if idx <= 11:   # bottom half, stacks upwards
        return x, BY + BOARD_H - R - k * 2 * R
    return x, BY + R + k * 2 * R   # top half, stacks downwards


def bar_pos(player, k):
    """Returns the pixel centre of the k-th checker of 'player' on the bar."""
    x = BX + 6 * PW + BAR_W // 2
    mid = BY + BOARD_H // 2
    return (x, mid + R + 6 + k * 2 * R) if player == WHITE else (x, mid - R - 6 - k * 2 * R)


def hit_test(pos):
    """
    Converts a mouse position into a board target.
    Returns a point index (0-23), 'bar', 'off', or None if nothing was clicked.
    """
    x, y = pos
    if TRAY_X <= x <= TRAY_X + TRAY_W and BY <= y <= BY + BOARD_H:
        return "off"
    if not (BX <= x <= BX + BOARD_W and BY <= y <= BY + BOARD_H):
        return None
    rel = x - BX
    if 6 * PW <= rel < 6 * PW + BAR_W:
        return "bar"
    col = rel // PW if rel < 6 * PW else (rel - BAR_W) // PW
    col = int(min(col, 11))
    if y >= BY + BOARD_H // 2:      # bottom half: points 1-12
        return 11 - col
    return col + 12                # top half: points 13-24


def die_for_move(move, player, dice_left):
    """
    Works out which die value a move used, so the panel can grey it out.
    move      : (src, dest) tuple
    player    : who moved
    dice_left : dice not yet used this turn
    """
    src, dest = move
    if src == "bar":
        dist = 24 - dest if player == WHITE else dest + 1
    elif dest == "off":
        dist = src + 1 if player == WHITE else 24 - src
    else:
        dist = abs(dest - src)
    if dist in dice_left:
        return dist
    bigger = [d for d in dice_left if d > dist]   # bearing off with a higher die
    return min(bigger) if bigger else dice_left[0]


def ease_out(t):
    """Cubic ease-out: fast start, gentle landing (t runs from 0 to 1)."""
    return 1 - (1 - t) ** 3


# ---------------------------------------------------------------------------
# AI analysis for the "AI thinking" panel and the Hint button
# ---------------------------------------------------------------------------

def analyse_turns(state, player, turns, depth, beam):
    """
    Does exactly what expectiminimax_choice() in ai.py does, but also keeps the
    expected value of every candidate so the panel can show them.
    Returns (chosen_turn, ranked) where ranked is a list of (value, moves), best first.
    The chosen turn is always identical to the one expectiminimax_choice() would pick.
    Parameters:
      state: the GameState (board position) being examined
      player: WHITE (1) or BLACK (-1), whose checker or turn this is
      turns: list of (moves, resulting_state) legal turns
      depth: number of turns to look ahead
      beam: number of candidate turns to keep (None = keep all)
    """
    if not turns[0][0] or len(turns) == 1:
        return turns[0], []
    candidates = order_turns(turns, player, beam)
    scored = [(chance_node(turn[1], -player, depth - 1, player, beam), i, turn)
              for i, turn in enumerate(candidates)]
    # highest value first; on a tie the earlier candidate wins, like the strict '>' in ai.py
    scored.sort(key=lambda v: (-v[0], v[1]))
    return scored[0][2], [(value, turn[0]) for value, _, turn in scored]


def format_turn(moves):
    """Turns a list of moves into board notation, e.g. '13/7 8/7'."""
    return " ".join(format_move(m) for m in moves) if moves else "(no move)"


def format_value(value):
    """Formats an expected value for the panel ('win' / 'loss' for decided games)."""
    if value >= WIN_SCORE:
        return "win"
    if value <= -WIN_SCORE:
        return "loss"
    return f"{value:+.1f}"


# ---------------------------------------------------------------------------
# Sound effects (generated in code, so no audio files are needed)
# ---------------------------------------------------------------------------

RATE = 22050


def make_sound(samples):
    """
    Turns a list of floats (-1..1) into a pygame Sound.
    The raw samples are converted to whatever format the mixer is running at
    (sample rate, bit depth, channels), so no WAV/OGG decoder is needed.
    This matters in the browser, where the mixer picks its own format.
    Parameters:
      samples: the audio waveform, RATE samples per second
    """
    freq, size, channels = pygame.mixer.get_init()
    n_out = max(1, int(len(samples) * freq / RATE))
    step = RATE / freq
    out = []
    for i in range(n_out):
        v = samples[min(len(samples) - 1, int(i * step))]
        v = max(-1.0, min(1.0, v))
        out.extend([v] * channels)
    if abs(size) == 32:            # float32 samples (common in browsers)
        data = array("f", out)
    elif abs(size) == 8:
        data = array("B" if size > 0 else "b",
                     (int(v * 127) + (128 if size > 0 else 0) for v in out))
    else:                          # 16-bit
        data = array("H" if size > 0 else "h",
                     (int(v * 30000) + (32768 if size > 0 else 0) for v in out))
    return pygame.mixer.Sound(buffer=data.tobytes())


def clack(length, freq, noise, decay, rng):
    """A short percussive 'clack': a decaying tone mixed with a burst of noise."""
    n = int(RATE * length)
    return [math.exp(-decay * i / RATE) *
            ((1 - noise) * math.sin(2 * math.pi * freq * i / RATE) + noise * rng.uniform(-1, 1))
            for i in range(n)]


def notes(freqs, each, volume=0.5):
    """A short melody: each frequency in 'freqs' is played for 'each' seconds."""
    out = []
    for f in freqs:
        n = int(RATE * each)
        out += [volume * math.exp(-4 * i / n) * math.sin(2 * math.pi * f * i / RATE)
                for i in range(n)]
    return out


class Sounds:
    """Holds the game's sound effects. Silently does nothing if audio is unavailable."""

    def __init__(self):
        self.enabled = True
        self.bank = {}
        try:
            if not pygame.mixer.get_init():
                if IN_BROWSER:
                    pygame.mixer.init()   # let the browser pick its own sample rate
                else:
                    pygame.mixer.init(RATE, -16, 1, 512)
            rng = random.Random(7)
            dice = [0.0] * int(RATE * 0.38)
            for start in (0.0, 0.07, 0.15, 0.24, 0.31):   # dice rattling in the cup
                for i, v in enumerate(clack(0.05, rng.randint(1800, 2600), 0.7, 90, rng)):
                    j = int(start * RATE) + i
                    if j < len(dice):
                        dice[j] += 0.5 * v
            self.bank = {
                "dice": make_sound(dice),
                "move": make_sound([0.6 * v for v in clack(0.07, 220, 0.45, 70, rng)]),
                "hit": make_sound([0.8 * v for v in clack(0.18, 110, 0.35, 22, rng)]),
                "off": make_sound([0.5 * v for v in clack(0.09, 880, 0.2, 45, rng)]),
                "win": make_sound(notes([523, 659, 784, 1047], 0.13)),
                "lose": make_sound(notes([392, 330, 262], 0.18)),
            }
        except Exception:     # no audio device (or unsupported in this browser)
            self.bank = {}

    def play(self, name):
        """Plays the named sound effect if sound is switched on."""
        if self.enabled and name in self.bank:
            try:
                self.bank[name].play()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Small UI widgets
# ---------------------------------------------------------------------------

class Button:
    """A clickable rectangle with a label. 'enabled' greys it out when False."""

    def __init__(self, rect, text, color=(70, 110, 200)):
        """
        Creates a clickable button.
        Parameters:
          rect: (x, y, width, height) of the button
          text: the label shown on the button / message text
          color: RGB fill colour of the button
        """
        self.rect = pygame.Rect(rect)
        self.text = text
        self.color = color
        self.enabled = True

    def draw(self, surf, font):
        """Draws the button, lighter when the mouse hovers over it.
        Parameters:
          surf: the pygame Surface to draw on
          font: pygame Font used for any text
        """
        hover = self.enabled and self.rect.collidepoint(pygame.mouse.get_pos())
        c = self.color if self.enabled else (70, 74, 82)
        if hover:
            c = tuple(min(255, v + 30) for v in c)
        pygame.draw.rect(surf, c, self.rect, border_radius=10)
        label = font.render(self.text, True, TEXT if self.enabled else MUTED)
        surf.blit(label, label.get_rect(center=self.rect.center))

    def clicked(self, pos):
        """True if the button is enabled and 'pos' is inside it."""
        return self.enabled and self.rect.collidepoint(pos)


def draw_checker(surf, pos, player, ring=None, label=None, font=None):
    """
    Draws one checker at 'pos'.
    ring  : optional colour for a highlight ring (selection / legal source)
    label : optional number drawn on the checker (for stacks taller than 5)
    Parameters:
      surf: the pygame Surface to draw on
      pos: (x, y) pixel position
      player: WHITE (1) or BLACK (-1), whose checker or turn this is
      ring: optional colour for a highlight ring around the checker
      label: optional text drawn on the checker (e.g. stack count)
      font: pygame Font used for any text
    """
    fill, edge = (WHITE_CHK, (170, 165, 150)) if player == WHITE else (BLACK_CHK, (90, 90, 100))
    pygame.draw.circle(surf, (0, 0, 0), (pos[0] + 2, pos[1] + 3), R)        # shadow
    pygame.draw.circle(surf, fill, pos, R)
    pygame.draw.circle(surf, edge, pos, R, 3)
    pygame.draw.circle(surf, edge, pos, R - 9, 2)
    if ring:
        pygame.draw.circle(surf, ring, pos, R + 3, 4)
    if label and font:
        t = font.render(label, True, BLACK_CHK if player == WHITE else WHITE_CHK)
        surf.blit(t, t.get_rect(center=pos))


def draw_die(surf, x, y, value, used=False, size=52, angle=0):
    """Draws a die face showing 'value' at (x, y). Used dice are drawn faded.
    Parameters:
      surf: the pygame Surface to draw on
      x: left pixel of the die
      y: top pixel of the die
      value: the number (1-6) shown on the die
      used: True if this die has already been played (drawn faded)
      size: width and height of the die in pixels
      angle: rotation in degrees (used while the dice are tumbling)
    """
    face = pygame.Surface((size, size), pygame.SRCALPHA)
    rect = pygame.Rect(0, 0, size, size)
    pygame.draw.rect(face, (120, 120, 120) if used else (250, 250, 250), rect, border_radius=9)
    pygame.draw.rect(face, (40, 40, 40), rect, 2, border_radius=9)
    s, c = size // 4, size // 2
    spots = {1: [(c, c)], 2: [(s, s), (3 * s, 3 * s)], 3: [(s, s), (c, c), (3 * s, 3 * s)],
             4: [(s, s), (3 * s, s), (s, 3 * s), (3 * s, 3 * s)],
             5: [(s, s), (3 * s, s), (c, c), (s, 3 * s), (3 * s, 3 * s)],
             6: [(s, s), (3 * s, s), (s, c), (3 * s, c), (s, 3 * s), (3 * s, 3 * s)]}
    for dx, dy in spots[value]:
        pygame.draw.circle(face, (30, 30, 30), (dx, dy), size // 11)
    if angle:
        face = pygame.transform.rotate(face, angle)
    surf.blit(face, face.get_rect(center=(x + size // 2, y + size // 2)))


def top_pos(state, player, where):
    """
    Returns the pixel centre of 'player's top checker at 'where' (a point index, 'bar' or 'off').
    Used as the start and end points of the sliding-checker animation and the hint arrows.
    Parameters:
      state: the GameState (board position) being examined
      player: WHITE (1) or BLACK (-1), whose checker or turn this is
      where: point index 0-23, "bar" or "off"
    """
    if where == "bar":
        return bar_pos(player, max(min(state.bar[player], 4) - 1, 0))
    if where == "off":
        k = max(state.off[player] - 1, 0)
        y = BY + BOARD_H - 7 - k * 14 if player == WHITE else BY + 7 + k * 14
        return TRAY_X + TRAY_W // 2, y
    return checker_pos(where, max(min(state.count_on(where, player), 5) - 1, 0))


def draw_arrow(surf, start, end, colour, width=5):
    """Draws a straight arrow from 'start' to 'end' (used to show the hint)."""
    pygame.draw.line(surf, colour, start, end, width)
    ang = math.atan2(end[1] - start[1], end[0] - start[0])
    head = [end,
            (end[0] - 18 * math.cos(ang - 0.45), end[1] - 18 * math.sin(ang - 0.45)),
            (end[0] - 18 * math.cos(ang + 0.45), end[1] - 18 * math.sin(ang + 0.45))]
    pygame.draw.polygon(surf, colour, head)


# ---------------------------------------------------------------------------
# The game application
# ---------------------------------------------------------------------------

class BackgammonApp:
    """Holds the game state and runs the menu, the game loop and all drawing."""

    def __init__(self):
        if not IN_BROWSER:
            pygame.mixer.pre_init(RATE, -16, 1, 512)
        pygame.init()
        pygame.display.set_caption(f"Backgammon AI - {STUDENT_NAME} ({STUDENT_ID})")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock = pygame.time.Clock()
        self.f_big = pygame.font.Font(None, 84)
        self.f_mid = pygame.font.Font(None, 40)
        self.f_small = pygame.font.Font(None, 28)
        self.f_tiny = pygame.font.Font(None, 22)
        self.f_mini = pygame.font.Font(None, 19)
        self.sounds = Sounds()

        self.phase = "menu"
        cx = WIN_W // 2
        self.level_buttons = [Button((cx - 330 + i * 230, 400, 200, 64), name, col)
                              for i, (name, col) in enumerate([("Easy", (60, 150, 90)),
                                                                ("Medium", (200, 140, 40)),
                                                                ("Hard", (190, 60, 60))])]
        self.roll_btn = Button((PANEL_X, 212, 62, 40), "Roll")
        self.undo_btn = Button((PANEL_X + 69, 212, 62, 40), "Undo", (110, 110, 130))
        self.hint_btn = Button((PANEL_X + 138, 212, 62, 40), "Hint", (40, 140, 160))
        self.sound_btn = Button((WIN_W - 112, WIN_H - 54, 84, 26), "Sound on", (70, 80, 96))
        self.again_btn = Button((cx - 220, 430, 200, 60), "Play again", (60, 150, 90))
        self.menu_btn = Button((cx + 20, 430, 200, 60), "Menu", (110, 110, 130))

    # ----- game flow --------------------------------------------------------

    def new_game(self, level):
        """Resets everything and starts a game against the chosen AI 'level'."""
        self.level = level
        self.ai_fn = AI_LEVELS[level]
        self.state = GameState()
        self.dice, self.dice_left = [], []
        self.selected = None
        self.played, self.sequences = [], []
        self.last_ai_dests = []
        self.log = []
        self.anims = []            # checkers currently sliding across the board
        self.analysis = None       # what the "AI thinking" panel shows
        self.hint, self.hint_state, self.hint_job = None, None, None
        self.gammon = False
        self.current = random.choice([WHITE, BLACK])
        self.add_log(f"New game vs {level} AI.")
        self.add_log("You start." if self.current == WHITE else "The AI starts.")
        self.begin_turn()

    def add_log(self, text):
        """Adds a line to the message log in the side panel.
        Parameters:
          text: the label shown on the button / message text
        """
        self.log.append(text)
        self.log = self.log[-8:]

    def begin_turn(self):
        """Starts the turn of self.current: the human must press Roll, the AI rolls itself."""
        self.selected = None
        self.hint, self.hint_job = None, None
        self.dice, self.dice_left = [], []
        if self.current == WHITE:
            self.phase = "await_roll"
        else:
            self.start_roll()

    def start_roll(self):
        """Begins the dice-rolling animation for the current player."""
        self.dice = roll_dice()
        self.dice_left = list(self.dice)
        self.phase = "rolling"
        self.timer = pygame.time.get_ticks()
        self.roll_faces, self.face_timer = [random.randint(1, 6), random.randint(1, 6)], self.timer
        self.sounds.play("dice")

    def after_roll(self):
        """Called when the roll animation finishes: sets up the human turn or starts the AI."""
        who = "You" if self.current == WHITE else "AI"
        self.add_log(f"{who} rolled {self.dice[0]}-{self.dice[1]}.")
        if self.current == WHITE:
            self.turn_start = self.state
            self.played = []
            self.sequences = [m for m, _ in get_legal_turns(self.state, WHITE, self.dice, dedupe=False)]
            if not self.sequences[0]:
                self.add_log("No legal moves - you pass.")
                self.phase, self.timer = "pass", pygame.time.get_ticks()
            else:
                self.phase = "human_move"
        else:
            self.ai_result = None
            self.phase, self.timer = "ai_thinking", pygame.time.get_ticks()
            if not IN_BROWSER:   # desktop: think in the background; browser: think in update()
                threading.Thread(target=self.ai_worker, args=(self.state, list(self.dice)),
                                 daemon=True).start()

    def make_analysis(self, title, ranked, n_turns, seconds, depth, beam):
        """
        Packs search results into the form the side panel draws.
        Parameters:
          title: heading shown above the list
          ranked: list of (value, moves), best first, from analyse_turns()
          n_turns: number of distinct legal turns there were
          seconds: time the search took
          depth: search depth used
          beam: beam width used (None = all turns)
        """
        if not ranked:
            return {"title": title, "rows": [], "notes": ["Only one legal turn."]}
        rows = [(format_turn(moves), format_value(value)) for value, moves in ranked[:3]]
        searched = f"{len(ranked)} of {n_turns} turns" if beam else f"all {n_turns} turns"
        how = "expected value (21 rolls)" if depth >= 2 else "score after the move"
        return {"title": title, "rows": rows,
                "notes": [f"{searched}, {seconds * 1000:.0f} ms", how]}

    def ai_worker(self, state, dice):
        """Runs the AI search in a background thread so the window never freezes.
        Parameters:
          state: the GameState (board position) being examined
          dice: list of dice values still to play (4 copies on a double)
        """
        turns = get_legal_turns(state, BLACK, dice)
        start = time.time()
        if self.level in AI_SEARCH:
            depth, beam = AI_SEARCH[self.level]
            (moves, _), ranked = analyse_turns(state, BLACK, turns, depth, beam)
            title = f"{self.level} AI (depth {depth})"
            analysis = self.make_analysis(title, ranked, len(turns), time.time() - start, depth, beam)
        else:
            moves, _ = self.ai_fn(state, BLACK, dice, turns)
            analysis = {"title": "Easy AI (random)", "rows": [(format_turn(moves), "")],
                        "notes": [f"picked at random from {len(turns)} turns"]}
        if not moves:
            analysis = {"title": analysis["title"], "rows": [], "notes": ["No legal move - pass."]}
        elif not analysis["rows"]:
            analysis["rows"] = [(format_turn(moves), "")]
        self.ai_analysis = analysis
        self.ai_result = moves          # set last: the main loop waits for this

    def end_turn(self):
        """Checks for a winner, otherwise hands the turn to the other player."""
        win = self.state.winner()
        if win is not None:
            loser = -win
            self.gammon = self.state.off[loser] == 0
            self.phase = "gameover"
            self.sounds.play("win" if win == WHITE else "lose")
            return
        self.current = -self.current
        self.begin_turn()

    # ----- hints -------------------------------------------------------------

    def request_hint(self):
        """Asks the Hard AI for the best way to play the rest of the human's turn."""
        if self.phase != "human_move" or self.hint_job is not None:
            return
        if self.hint and self.hint_state is self.state:
            return                      # already showing a hint for this position
        k = len(self.played)
        unique = {}
        for seq in self.sequences:
            if len(seq) > k and [tuple(m) for m in seq[:k]] == self.played:
                rest = [tuple(m) for m in seq[k:]]
                result = self.state
                for m in rest:
                    result = apply_move(result, WHITE, m)
                unique.setdefault(result.key(), (rest, result))
        if not unique:
            return
        self.hint_job = {"state": self.state, "turns": list(unique.values()),
                         "t": pygame.time.get_ticks(), "started": False, "result": None}
        if not IN_BROWSER:
            self.hint_job["started"] = True
            threading.Thread(target=self.hint_worker, args=(self.hint_job,), daemon=True).start()

    def hint_worker(self, job):
        """Searches the human's options with the Hard AI's settings.
        Parameters:
          job: dict holding the position and candidate turns; the result is stored in it
        """
        depth, beam = HINT_SEARCH
        start = time.time()
        (moves, _), ranked = analyse_turns(job["state"], WHITE, job["turns"], depth, beam)
        analysis = self.make_analysis("Hint (Hard AI for you)", ranked, len(job["turns"]),
                                      time.time() - start, depth, beam)
        if not ranked:
            analysis["rows"] = [(format_turn(moves), "")]
        job["result"] = (moves, analysis)

    # ----- human move handling ---------------------------------------------

    def legal_next_moves(self):
        """Returns the set of single moves the human may make next, given moves already played."""
        k = len(self.played)
        return {tuple(seq[k]) for seq in self.sequences
                if len(seq) > k and [tuple(m) for m in seq[:k]] == self.played}

    def play_move(self, player, move, anim_ms):
        """Applies one single-checker move to the board, marks its die as used and animates it.
        Parameters:
          player: WHITE (1) or BLACK (-1), whose checker or turn this is
          move: a (src, dest) tuple for one checker move
          anim_ms: how long the checker takes to slide to its new place
        Returns True if the move hit an opponent checker.
        """
        before = self.state
        self.dice_left.remove(die_for_move(move, player, self.dice_left))
        self.state = apply_move(self.state, player, move)
        src, dest = move
        hit = dest != "off" and before.points[dest] == -player
        now = pygame.time.get_ticks()
        self.anims.append({"player": player, "frm": top_pos(before, player, src),
                           "to": top_pos(self.state, player, dest), "start": now, "dur": anim_ms,
                           "hide": ("off", player) if dest == "off" else ("point", dest),
                           "sound": "hit" if hit else ("off" if dest == "off" else "move")})
        if hit:   # the hit checker is knocked onto the bar just after the attacker lands
            self.anims.append({"player": -player, "frm": checker_pos(dest, 0),
                               "to": top_pos(self.state, -player, "bar"),
                               "start": now + int(anim_ms * 0.7), "dur": anim_ms,
                               "hide": ("bar", -player), "sound": None})
        return hit

    def handle_board_click(self, target):
        """Select a checker, move the selected checker, or change/clear the selection.
        Parameters:
          target: the point index, "bar" or "off" that was clicked
        """
        nxt = self.legal_next_moves()
        sources = {m[0] for m in nxt}
        if self.selected is not None and (self.selected, target) in nxt:
            move = (self.selected, target)
            follows_hint = bool(self.hint) and self.hint_state is self.state and self.hint[0] == move
            self.play_move(WHITE, move, HUMAN_ANIM_MS)
            self.played.append(move)
            self.selected = None
            if follows_hint:           # keep showing the rest of the suggested turn
                self.hint, self.hint_state = self.hint[1:], self.state
            if not self.legal_next_moves():
                self.end_turn()
        elif target in sources and target != self.selected:
            self.selected = target
        else:
            self.selected = None

    def undo(self):
        """Takes back all moves made so far this turn."""
        self.state = self.turn_start
        self.played = []
        self.dice_left = list(self.dice)
        self.selected = None
        self.anims = []

    # ----- main loop ---------------------------------------------------------

    def update(self):
        """Advances timers: dice animation, pass delays, AI move animation and hints."""
        now = pygame.time.get_ticks()

        # Finished checker slides: play their landing sound and forget them
        for a in [a for a in self.anims if now >= a["start"] + a["dur"]]:
            self.anims.remove(a)
            if a["sound"]:
                self.sounds.play(a["sound"])

        # Hints (in the browser the search runs here, one frame after the button press)
        job = self.hint_job
        if job is not None:
            if not job["started"] and now - job["t"] > 60:
                job["started"] = True
                self.hint_worker(job)
            if job["result"] is not None:
                self.hint_job = None
                if job["state"] is self.state and self.phase == "human_move":
                    self.hint, self.analysis = job["result"]
                    self.hint_state = self.state
                    self.add_log(f"Hint: {format_turn(self.hint)}")

        if self.phase == "rolling":
            if now - self.face_timer > 70:     # tumbling dice show a new random face
                self.roll_faces = [random.randint(1, 6), random.randint(1, 6)]
                self.face_timer = now
            if now - self.timer > ROLL_ANIM_MS:
                self.after_roll()
        elif self.phase == "pass" and now - self.timer > 1400:
            self.end_turn()
        elif self.phase == "ai_thinking" and IN_BROWSER and self.ai_result is None \
                and now - self.timer > 150:
            self.ai_worker(self.state, list(self.dice))
        elif self.phase == "ai_thinking" and self.ai_result is not None and now - self.timer > 500:
            self.ai_queue = list(self.ai_result)
            self.analysis = self.ai_analysis
            self.last_ai_dests = []
            if not self.ai_queue:
                self.add_log("AI has no legal moves.")
            self.phase, self.timer = "ai_moving", now
        elif self.phase == "ai_moving" and now - self.timer > AI_MOVE_MS:
            if self.ai_queue:
                move = self.ai_queue.pop(0)
                if self.play_move(BLACK, move, MOVE_ANIM_MS):
                    self.add_log("AI hit your checker!")
                self.last_ai_dests.append(move[1])
                self.timer = now
            elif not self.anims:
                self.end_turn()

    def handle_click(self, pos):
        """Routes a mouse click depending on the current screen/phase.
        Parameters:
          pos: (x, y) pixel position
        """
        if self.phase == "menu":
            for b in self.level_buttons:
                if b.clicked(pos):
                    self.new_game(b.text)
        elif self.sound_btn.clicked(pos):
            self.toggle_sound()
        elif self.phase == "gameover":
            if self.again_btn.clicked(pos):
                self.new_game(self.level)
            elif self.menu_btn.clicked(pos):
                self.phase = "menu"
        elif self.phase == "await_roll" and self.roll_btn.clicked(pos):
            self.start_roll()
        elif self.phase == "human_move":
            if self.undo_btn.clicked(pos):
                self.undo()
            elif self.hint_btn.clicked(pos):
                self.request_hint()
            else:
                target = hit_test(pos)
                if target is not None:
                    self.handle_board_click(target)

    def toggle_sound(self):
        """Switches the sound effects on or off."""
        self.sounds.enabled = not self.sounds.enabled
        self.sound_btn.text = "Sound on" if self.sounds.enabled else "Sound off"

    def step(self):
        """One frame: handle events, update timers, draw. Returns False when the window is closed."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.handle_click(event.pos)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and self.phase == "await_roll":
                    self.start_roll()
                elif event.key == pygame.K_h and self.phase == "human_move":
                    self.request_hint()
                elif event.key == pygame.K_m:
                    self.toggle_sound()
        if self.phase != "menu":
            self.update()
        self.draw()
        pygame.display.flip()
        return True

    async def run(self):
        """
        The main loop at 60 frames per second. It is 'async' so the browser version
        (pygbag) can hand control back to the web page every frame via asyncio.sleep(0).
        """
        while self.step():
            self.clock.tick(60)
            await asyncio.sleep(0)
        pygame.quit()

    # ----- drawing -----------------------------------------------------------

    def fit(self, font, text, width, colour):
        """Renders 'text', shortening it with '..' if it is wider than 'width' pixels."""
        img = font.render(text, True, colour)
        while img.get_width() > width and len(text) > 3:
            text = text[:-3] + ".."
            img = font.render(text, True, colour)
        return img

    def draw(self):
        """Draws whichever screen is active."""
        self.screen.fill(BG)
        if self.phase == "menu":
            self.draw_menu()
            return
        self.draw_board()
        self.draw_panel()
        if self.phase == "gameover":
            self.draw_gameover()

    def draw_menu(self):
        """Title screen with student details and difficulty buttons."""
        s = self.screen
        s.fill(FELT)
        t = self.f_big.render("BACKGAMMON", True, TEXT)
        s.blit(t, t.get_rect(center=(WIN_W // 2, 170)))
        t = self.f_mid.render("vs an Expectiminimax AI", True, GOLD)
        s.blit(t, t.get_rect(center=(WIN_W // 2, 235)))
        t = self.f_small.render(f"{STUDENT_NAME}   |   Student ID: {STUDENT_ID}", True, TEXT)
        s.blit(t, t.get_rect(center=(WIN_W // 2, 290)))
        t = self.f_small.render("Choose a difficulty:", True, MUTED)
        s.blit(t, t.get_rect(center=(WIN_W // 2, 365)))
        for b in self.level_buttons:
            b.draw(s, self.f_mid)
        tips = ["Easy: random moves     Medium: greedy (depth 1)     Hard: Expectiminimax (depth 2)",
                "You play White. Click a checker, then a green marker.",
                "Space = roll     H = hint     M = sound on/off"]
        for i, line in enumerate(tips):
            t = self.f_tiny.render(line, True, MUTED)
            s.blit(t, t.get_rect(center=(WIN_W // 2, 515 + i * 28)))
        t = self.f_tiny.render("5ENT1140 Artificial Intelligence Principles - Mini Project", True, MUTED)
        s.blit(t, t.get_rect(center=(WIN_W // 2, WIN_H - 30)))

    def draw_board(self):
        """Draws the frame, triangles, bar, checkers, bear-off tray and highlights."""
        s = self.screen
        st = self.state
        now = pygame.time.get_ticks()
        pygame.draw.rect(s, FRAME, (BX - 14, BY - 14, BOARD_W + 28 + TRAY_W + 14, BOARD_H + 28),
                         border_radius=12)
        pygame.draw.rect(s, FELT, (BX, BY, BOARD_W, BOARD_H))
        pygame.draw.rect(s, FRAME_DARK, (BX + 6 * PW, BY, BAR_W, BOARD_H))
        pygame.draw.rect(s, (40, 30, 20), (TRAY_X, BY, TRAY_W, BOARD_H), border_radius=6)

        # Triangles and point numbers
        for idx in range(24):
            cx = point_center_x(idx)
            col = point_column(idx)
            colour = POINT_A if col % 2 == 0 else POINT_B
            if idx <= 11:
                base, apex, ny = BY + BOARD_H, BY + BOARD_H - TRI_H, BY + BOARD_H + 22
            else:
                base, apex, ny = BY, BY + TRI_H, BY - 22
            pygame.draw.polygon(s, colour, [(cx - PW // 2 + 2, base), (cx + PW // 2 - 2, base), (cx, apex)])
            n = self.f_tiny.render(str(idx + 1), True, MUTED)
            s.blit(n, n.get_rect(center=(cx, ny)))

        # Work out highlights for the human's turn
        sources, dests = set(), set()
        if self.phase == "human_move":
            nxt = self.legal_next_moves()
            sources = {m[0] for m in nxt}
            if self.selected is not None:
                dests = {m[1] for m in nxt if m[0] == self.selected}

        # Checkers still sliding are drawn separately, so leave them out of their new place
        hidden = {}
        for a in self.anims:
            hidden[a["hide"]] = hidden.get(a["hide"], 0) + 1

        # Checkers on the points
        for idx in range(24):
            n = abs(st.points[idx]) - hidden.get(("point", idx), 0)
            if n <= 0:
                continue
            player = WHITE if st.points[idx] > 0 else BLACK
            shown = min(n, 5)
            for k in range(shown):
                top = k == shown - 1
                ring = None
                if top and player == WHITE and idx in sources:
                    ring = GOLD if idx == self.selected else (200, 170, 90)
                if top and player == BLACK and idx in self.last_ai_dests:
                    ring = (90, 170, 255)
                label = str(n) if top and n > 5 else None
                draw_checker(s, checker_pos(idx, k), player, ring, label, self.f_small)

        # Checkers on the bar
        for p in (WHITE, BLACK):
            count = st.bar[p] - hidden.get(("bar", p), 0)
            for k in range(min(count, 4)):
                ring = None
                if p == WHITE and "bar" in sources and k == 0:
                    ring = GOLD if self.selected == "bar" else (200, 170, 90)
                label = str(count) if k == 0 and count > 1 else None
                draw_checker(s, bar_pos(p, k), p, ring, label, self.f_small)

        # Borne-off checkers in the tray (thin slabs)
        for k in range(st.off[WHITE] - hidden.get(("off", WHITE), 0)):
            pygame.draw.rect(s, WHITE_CHK, (TRAY_X + 6, BY + BOARD_H - 12 - k * 14, TRAY_W - 12, 11),
                             border_radius=3)
        for k in range(st.off[BLACK] - hidden.get(("off", BLACK), 0)):
            pygame.draw.rect(s, (70, 70, 78), (TRAY_X + 6, BY + 2 + k * 14, TRAY_W - 12, 11),
                             border_radius=3)

        # Legal destination markers
        for d in dests:
            if d == "off":
                pygame.draw.rect(s, GREEN, (TRAY_X, BY, TRAY_W, BOARD_H), 4, border_radius=6)
            else:
                k = min(abs(st.points[d]) if st.points[d] > 0 else 0, 4)
                pygame.draw.circle(s, GREEN, checker_pos(d, k), R - 4, 5)

        # Hint arrows: the Hard AI's suggested moves, numbered in the order to play them
        if self.hint and self.hint_state is st and self.phase == "human_move":
            layer = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
            pos_state = st
            for i, move in enumerate(self.hint):
                frm = top_pos(pos_state, WHITE, move[0])
                pos_state = apply_move(pos_state, WHITE, move)
                to = top_pos(pos_state, WHITE, move[1])
                draw_arrow(layer, frm, to, CYAN + (190,))
                pygame.draw.circle(layer, CYAN + (230,), frm, 11)
                num = self.f_tiny.render(str(i + 1), True, BG)
                layer.blit(num, num.get_rect(center=frm))
            s.blit(layer, (0, 0))

        # Checkers in flight: ease towards the target with a small hop
        for a in self.anims:
            t = (now - a["start"]) / a["dur"]
            if t <= 0:
                pos = a["frm"]
            else:
                t = min(t, 1.0)
                e = ease_out(t)
                pos = (int(a["frm"][0] + (a["to"][0] - a["frm"][0]) * e),
                       int(a["frm"][1] + (a["to"][1] - a["frm"][1]) * e - math.sin(math.pi * t) * 30))
            draw_checker(s, pos, a["player"])

    def draw_panel(self):
        """Side panel: turn info, dice, buttons, pip counts, AI analysis and message log."""
        s = self.screen
        x = PANEL_X
        w = WIN_W - 28 - x                  # usable width inside the panel
        now = pygame.time.get_ticks()
        pygame.draw.rect(s, PANEL, (x - 14, 20, WIN_W - x - 6, WIN_H - 40), border_radius=12)
        s.blit(self.f_mid.render("Backgammon", True, TEXT), (x, 36))
        s.blit(self.f_tiny.render(f"vs {self.level} AI", True, GOLD), (x, 72))

        dots = "." * (1 + (now // 300) % 3)
        if self.phase in ("ai_thinking", "ai_moving") or (self.phase == "rolling" and self.current == BLACK):
            turn_text = f"AI is thinking{dots}" if self.phase == "ai_thinking" else "AI's turn"
        elif self.phase == "gameover":
            turn_text = "Game over"
        else:
            turn_text = "Your turn"
        s.blit(self.f_small.render(turn_text, True, TEXT), (x, 102))

        # Dice: they tumble and bounce while rolling, then settle
        if self.dice:
            if self.phase == "rolling":
                t = min((now - self.timer) / ROLL_ANIM_MS, 1.0)
                for i, f in enumerate(self.roll_faces):
                    hop = abs(math.sin(t * 3 * math.pi + i)) * (1 - t) * 24
                    spin = (1 - t) ** 2 * 540 * (1 if i == 0 else -1)
                    draw_die(s, x + 6 + i * 64, 138 - int(hop), f, False, 52, spin)
            else:
                faces = self.dice[:2] if len(self.dice) == 2 else self.dice
                left = list(self.dice_left)
                used = []
                for f in faces:
                    if f in left:
                        left.remove(f)
                        used.append(False)
                    else:
                        used.append(True)
                size = 52 if len(faces) == 2 else 40
                for i, f in enumerate(faces):
                    gap = 12 if size == 52 else 8
                    draw_die(s, x + 6 + i * (size + gap), 138 + (52 - size) // 2, f, used[i], size)

        self.roll_btn.enabled = self.phase == "await_roll"
        self.undo_btn.enabled = self.phase == "human_move" and bool(self.played)
        self.hint_btn.enabled = self.phase == "human_move" and self.hint_job is None
        if self.roll_btn.enabled:            # gentle pulse so it is obvious what to do next
            glow = int(3 + 2 * math.sin(now / 180))
            pygame.draw.rect(s, GOLD, self.roll_btn.rect.inflate(glow * 2, glow * 2), 2, border_radius=12)
        for b in (self.roll_btn, self.undo_btn, self.hint_btn):
            b.draw(s, self.f_small)

        s.blit(self.f_tiny.render(f"Your pips: {pip_count(self.state, WHITE)}", True, TEXT), (x, 266))
        s.blit(self.f_tiny.render(f"AI pips:   {pip_count(self.state, BLACK)}", True, TEXT), (x, 288))
        s.blit(self.f_tiny.render(f"Off: you {self.state.off[WHITE]}  AI {self.state.off[BLACK]}",
                                  True, TEXT), (x, 310))

        self.draw_analysis(x, 340, w)

        s.blit(self.f_tiny.render("Log", True, MUTED), (x, 504))
        for i, line in enumerate(self.log[-5:]):
            s.blit(self.fit(self.f_tiny, line, w, TEXT), (x, 526 + i * 22))

        s.blit(self.f_tiny.render(STUDENT_NAME, True, MUTED), (x, WIN_H - 70))
        s.blit(self.f_tiny.render(f"ID: {STUDENT_ID}", True, MUTED), (x, WIN_H - 48))
        self.sound_btn.draw(s, self.f_mini)

    def draw_analysis(self, x, y, w):
        """
        The "AI thinking" box: the top 3 candidate turns the search found and their values.
        Parameters:
          x: left pixel of the box contents
          y: top pixel of the box
          w: usable width in pixels
        """
        s = self.screen
        box = pygame.Rect(x - 6, y, w + 12, 152)
        pygame.draw.rect(s, (24, 29, 36), box, border_radius=8)
        now = pygame.time.get_ticks()
        searching = (self.phase == "ai_thinking" and self.level in AI_SEARCH) or self.hint_job is not None
        if searching:
            who = "Hint: searching" if self.hint_job is not None else "Searching"
            s.blit(self.f_tiny.render(who + "." * (1 + (now // 300) % 3), True, CYAN), (x, y + 8))
            depth, _ = HINT_SEARCH if self.hint_job is not None else AI_SEARCH[self.level]
            lines = (["MAX: each candidate turn", "CHANCE: all 21 dice rolls", "MIN: opponent's best reply"]
                     if depth >= 2 else ["Scoring each turn with", "the evaluation function"])
            for i, line in enumerate(lines):
                s.blit(self.f_mini.render(line, True, MUTED), (x, y + 36 + i * 22))
            # a small bar sweeping back and forth while the search runs
            sweep = (math.sin(now / 250) + 1) / 2
            pygame.draw.rect(s, (50, 58, 70), (x, y + 128, w, 6), border_radius=3)
            pygame.draw.rect(s, CYAN, (x + int(sweep * (w - 50)), y + 128, 50, 6), border_radius=3)
            return
        a = self.analysis
        if a is None:
            s.blit(self.f_tiny.render("AI thinking", True, GOLD), (x, y + 8))
            for i, line in enumerate(["The AI's top candidate moves", "appear here after its turn.",
                                      "Press Hint to ask it for yours."]):
                s.blit(self.f_mini.render(line, True, MUTED), (x, y + 36 + i * 22))
            return
        colour = CYAN if a["title"].startswith("Hint") else GOLD
        s.blit(self.fit(self.f_tiny, a["title"], w, colour), (x, y + 8))
        for i, (moves, value) in enumerate(a["rows"]):
            ry = y + 34 + i * 22
            if i == 0:
                pygame.draw.rect(s, (40, 60, 52), (x - 4, ry - 3, w + 8, 21), border_radius=4)
            val = self.f_mini.render(value, True, GREEN if i == 0 else TEXT)
            s.blit(val, (x + w - val.get_width(), ry))
            s.blit(self.fit(self.f_mini, f"{i + 1}. {moves}", w - val.get_width() - 8,
                            TEXT if i == 0 else MUTED), (x, ry))
        for i, note in enumerate(a["notes"]):
            s.blit(self.fit(self.f_mini, note, w, MUTED), (x, y + 104 + i * 20))

    def draw_gameover(self):
        """Dark overlay announcing the winner, with Play again / Menu buttons."""
        s = self.screen
        overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        s.blit(overlay, (0, 0))
        won = self.state.winner() == WHITE
        t = self.f_big.render("You win!" if won else "The AI wins!", True, GREEN if won else (240, 90, 90))
        s.blit(t, t.get_rect(center=(WIN_W // 2, 300)))
        if self.gammon:
            t = self.f_mid.render("Gammon! (loser bore off no checkers)", True, GOLD)
            s.blit(t, t.get_rect(center=(WIN_W // 2, 370)))
        self.again_btn.draw(s, self.f_small)
        self.menu_btn.draw(s, self.f_small)


if __name__ == "__main__":
    asyncio.run(BackgammonApp().run())
