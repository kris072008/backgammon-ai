"""
Backgammon with Expectiminimax AI - Graphical Interface (Step 3)
Name:       Krishna Mahajan
Student ID: 106wucww
Module:     5ENT1140 Artificial Intelligence Principles - Mini Project

Run with:  python3 main.py   (or python3 gui.py)
You play White (moving from point 24 down to 1). The AI plays Black.
Click one of your checkers, then click a highlighted destination.
"""

import sys
import random
import asyncio
import threading
import pygame

from backgammon import WHITE, BLACK, GameState, roll_dice, get_legal_turns, apply_move
from ai import easy_ai, medium_ai, hard_ai, pip_count

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
WHITE_CHK = (242, 238, 228)
BLACK_CHK = (34, 34, 38)

AI_LEVELS = {"Easy": easy_ai, "Medium": medium_ai, "Hard": hard_ai}
ROLL_ANIM_MS = 600
AI_MOVE_MS = 550


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


def draw_die(surf, x, y, value, used=False, size=52):
    """Draws a die face showing 'value' at (x, y). Used dice are drawn faded.
    Parameters:
      surf: the pygame Surface to draw on
      x: left pixel of the die
      y: top pixel of the die
      value: the number (1-6) shown on the die
      used: True if this die has already been played (drawn faded)
      size: width and height of the die in pixels
    """
    rect = pygame.Rect(x, y, size, size)
    pygame.draw.rect(surf, (120, 120, 120) if used else (250, 250, 250), rect, border_radius=9)
    pygame.draw.rect(surf, (40, 40, 40), rect, 2, border_radius=9)
    s, c = size // 4, size // 2
    spots = {1: [(c, c)], 2: [(s, s), (3 * s, 3 * s)], 3: [(s, s), (c, c), (3 * s, 3 * s)],
             4: [(s, s), (3 * s, s), (s, 3 * s), (3 * s, 3 * s)],
             5: [(s, s), (3 * s, s), (c, c), (s, 3 * s), (3 * s, 3 * s)],
             6: [(s, s), (3 * s, s), (s, c), (3 * s, c), (s, 3 * s), (3 * s, 3 * s)]}
    for dx, dy in spots[value]:
        pygame.draw.circle(surf, (30, 30, 30), (x + dx, y + dy), size // 11)


# ---------------------------------------------------------------------------
# The game application
# ---------------------------------------------------------------------------

class BackgammonApp:
    """Holds the game state and runs the menu, the game loop and all drawing."""

    def __init__(self):
        pygame.init()
        pygame.display.set_caption(f"Backgammon AI - {STUDENT_NAME} ({STUDENT_ID})")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock = pygame.time.Clock()
        self.f_big = pygame.font.Font(None, 84)
        self.f_mid = pygame.font.Font(None, 40)
        self.f_small = pygame.font.Font(None, 28)
        self.f_tiny = pygame.font.Font(None, 22)

        self.phase = "menu"
        cx = WIN_W // 2
        self.level_buttons = [Button((cx - 330 + i * 230, 400, 200, 64), name, col)
                              for i, (name, col) in enumerate([("Easy", (60, 150, 90)),
                                                                ("Medium", (200, 140, 40)),
                                                                ("Hard", (190, 60, 60))])]
        self.roll_btn = Button((PANEL_X, 250, 90, 46), "Roll")
        self.undo_btn = Button((PANEL_X + 102, 250, 90, 46), "Undo", (110, 110, 130))
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

    def ai_worker(self, state, dice):
        """Runs the AI search in a background thread so the window never freezes.
        Parameters:
          state: the GameState (board position) being examined
          dice: list of dice values still to play (4 copies on a double)
        """
        turns = get_legal_turns(state, BLACK, dice)
        moves, _ = self.ai_fn(state, BLACK, dice, turns)
        self.ai_result = moves

    def end_turn(self):
        """Checks for a winner, otherwise hands the turn to the other player."""
        win = self.state.winner()
        if win is not None:
            loser = -win
            self.gammon = self.state.off[loser] == 0
            self.phase = "gameover"
            return
        self.current = -self.current
        self.begin_turn()

    # ----- human move handling ---------------------------------------------

    def legal_next_moves(self):
        """Returns the set of single moves the human may make next, given moves already played."""
        k = len(self.played)
        return {tuple(seq[k]) for seq in self.sequences
                if len(seq) > k and [tuple(m) for m in seq[:k]] == self.played}

    def play_move(self, player, move):
        """Applies one single-checker move to the board and marks its die as used.
        Parameters:
          player: WHITE (1) or BLACK (-1), whose checker or turn this is
          move: a (src, dest) tuple for one checker move
        """
        self.dice_left.remove(die_for_move(move, player, self.dice_left))
        self.state = apply_move(self.state, player, move)

    def handle_board_click(self, target):
        """Select a checker, move the selected checker, or change/clear the selection.
        Parameters:
          target: the point index, "bar" or "off" that was clicked
        """
        nxt = self.legal_next_moves()
        sources = {m[0] for m in nxt}
        if self.selected is not None and (self.selected, target) in nxt:
            move = (self.selected, target)
            self.play_move(WHITE, move)
            self.played.append(move)
            self.selected = None
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

    # ----- main loop ---------------------------------------------------------

    def update(self):
        """Advances timers: dice animation, pass delays, and AI move animation."""
        now = pygame.time.get_ticks()
        if self.phase == "rolling" and now - self.timer > ROLL_ANIM_MS:
            self.after_roll()
        elif self.phase == "pass" and now - self.timer > 1400:
            self.end_turn()
        elif self.phase == "ai_thinking" and IN_BROWSER and self.ai_result is None \
                and now - self.timer > 150:
            self.ai_worker(self.state, list(self.dice))
        elif self.phase == "ai_thinking" and self.ai_result is not None and now - self.timer > 500:
            self.ai_queue = list(self.ai_result)
            self.last_ai_dests = []
            if not self.ai_queue:
                self.add_log("AI has no legal moves.")
            self.phase, self.timer = "ai_moving", now
        elif self.phase == "ai_moving" and now - self.timer > AI_MOVE_MS:
            if self.ai_queue:
                move = self.ai_queue.pop(0)
                hit = move[1] != "off" and self.state.points[move[1]] == WHITE
                self.play_move(BLACK, move)
                self.last_ai_dests.append(move[1])
                if hit:
                    self.add_log("AI hit your checker!")
                self.timer = now
            else:
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
            else:
                target = hit_test(pos)
                if target is not None:
                    self.handle_board_click(target)

    def step(self):
        """One frame: handle events, update timers, draw. Returns False when the window is closed."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.handle_click(event.pos)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE \
                    and self.phase == "await_roll":
                self.start_roll()
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
                "You play White. Click a checker, then a green marker. Space = roll."]
        for i, line in enumerate(tips):
            t = self.f_tiny.render(line, True, MUTED)
            s.blit(t, t.get_rect(center=(WIN_W // 2, 520 + i * 30)))
        t = self.f_tiny.render("5ENT1140 Artificial Intelligence Principles - Mini Project", True, MUTED)
        s.blit(t, t.get_rect(center=(WIN_W // 2, WIN_H - 30)))

    def draw_board(self):
        """Draws the frame, triangles, bar, checkers, bear-off tray and highlights."""
        s = self.screen
        st = self.state
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

        # Checkers on the points
        for idx in range(24):
            n = abs(st.points[idx])
            if n == 0:
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
            for k in range(min(st.bar[p], 4)):
                ring = None
                if p == WHITE and "bar" in sources and k == 0:
                    ring = GOLD if self.selected == "bar" else (200, 170, 90)
                label = str(st.bar[p]) if k == 0 and st.bar[p] > 1 else None
                draw_checker(s, bar_pos(p, k), p, ring, label, self.f_small)

        # Borne-off checkers in the tray (thin slabs)
        for k in range(st.off[WHITE]):
            pygame.draw.rect(s, WHITE_CHK, (TRAY_X + 6, BY + BOARD_H - 12 - k * 14, TRAY_W - 12, 11),
                             border_radius=3)
        for k in range(st.off[BLACK]):
            pygame.draw.rect(s, (70, 70, 78), (TRAY_X + 6, BY + 2 + k * 14, TRAY_W - 12, 11),
                             border_radius=3)

        # Legal destination markers
        for d in dests:
            if d == "off":
                pygame.draw.rect(s, GREEN, (TRAY_X, BY, TRAY_W, BOARD_H), 4, border_radius=6)
            else:
                k = min(abs(st.points[d]) if st.points[d] > 0 else 0, 4)
                pygame.draw.circle(s, GREEN, checker_pos(d, k), R - 4, 5)

    def draw_panel(self):
        """Side panel: turn info, dice, buttons, pip counts and message log."""
        s = self.screen
        x = PANEL_X
        pygame.draw.rect(s, PANEL, (x - 14, 20, WIN_W - x - 6, WIN_H - 40), border_radius=12)
        s.blit(self.f_mid.render("Backgammon", True, TEXT), (x, 36))
        s.blit(self.f_tiny.render(f"vs {self.level} AI", True, GOLD), (x, 72))

        if self.phase in ("ai_thinking", "ai_moving") or (self.phase == "rolling" and self.current == BLACK):
            turn_text = "AI is thinking..." if self.phase == "ai_thinking" else "AI's turn"
        elif self.phase == "gameover":
            turn_text = "Game over"
        else:
            turn_text = "Your turn"
        s.blit(self.f_small.render(turn_text, True, TEXT), (x, 110))

        # Dice (random faces while rolling)
        if self.dice:
            if self.phase == "rolling":
                faces = [random.randint(1, 6), random.randint(1, 6)]
                used = [False, False]
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
                draw_die(s, x + i * (size + 8), 150, f, used[i], size)

        self.roll_btn.enabled = self.phase == "await_roll"
        self.undo_btn.enabled = self.phase == "human_move" and bool(self.played)
        self.roll_btn.draw(s, self.f_small)
        self.undo_btn.draw(s, self.f_small)

        s.blit(self.f_tiny.render(f"Your pips: {pip_count(self.state, WHITE)}", True, TEXT), (x, 320))
        s.blit(self.f_tiny.render(f"AI pips:   {pip_count(self.state, BLACK)}", True, TEXT), (x, 344))
        s.blit(self.f_tiny.render(f"Off: you {self.state.off[WHITE]}  AI {self.state.off[BLACK]}",
                                  True, TEXT), (x, 368))

        s.blit(self.f_tiny.render("Log", True, MUTED), (x, 410))
        for i, line in enumerate(self.log):
            s.blit(self.f_tiny.render(line, True, TEXT), (x, 436 + i * 24))

        s.blit(self.f_tiny.render(STUDENT_NAME, True, MUTED), (x, WIN_H - 70))
        s.blit(self.f_tiny.render(f"ID: {STUDENT_ID}", True, MUTED), (x, WIN_H - 48))

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
