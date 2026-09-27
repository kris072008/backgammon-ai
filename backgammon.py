"""
Backgammon with Expectiminimax AI - Core Game Logic (Step 1)
Name:       Krishna Mahajan
Student ID: 106wucww
Module:     5ENT1140 Artificial Intelligence Principles - Mini Project

Board layout (indexes 0-23, shown to players as points 1-24):
  - Positive numbers = White checkers, negative numbers = Black checkers.
  - White moves from point 24 down to point 1; White's home board is points 1-6.
  - Black moves from point 1 up to point 24; Black's home board is points 19-24.
"""

import random

WHITE = 1
BLACK = -1
NUM_POINTS = 24
CHECKERS_PER_PLAYER = 15
PLAYER_NAMES = {WHITE: "White", BLACK: "Black"}


class GameState:
    """
    Stores one complete backgammon position.
    points : list of 24 ints (+ = White checkers, - = Black checkers)
    bar    : dict player -> number of that player's checkers on the bar
    off    : dict player -> number of that player's checkers borne off
    """

    def __init__(self, points=None, bar=None, off=None):
        """
        Creates a position (the standard start position if no points are given).
        Parameters:
          points: list of 24 ints (+White / -Black), or None for the start position
          bar: dict player -> checkers on the bar (None = empty)
          off: dict player -> checkers borne off (None = none)
        """
        # If no position is given, create the standard starting position
        if points is None:
            points = [0] * NUM_POINTS
            points[23], points[12], points[7], points[5] = 2, 5, 3, 5       # White
            points[0], points[11], points[16], points[18] = -2, -5, -3, -5  # Black
        self.points = points
        self.bar = bar if bar is not None else {WHITE: 0, BLACK: 0}
        self.off = off if off is not None else {WHITE: 0, BLACK: 0}

    def copy(self):
        """Returns an independent copy of this state (so the AI can explore safely)."""
        return GameState(self.points[:], dict(self.bar), dict(self.off))

    def key(self):
        """Returns a hashable summary of the state, used to remove duplicate positions."""
        return (tuple(self.points), self.bar[WHITE], self.bar[BLACK],
                self.off[WHITE], self.off[BLACK])

    def count_on(self, idx, player):
        """Returns how many of 'player's checkers are on point index 'idx'."""
        n = self.points[idx] * player
        return n if n > 0 else 0

    def is_open(self, idx, player):
        """True if 'player' may land on 'idx' (the opponent has fewer than 2 checkers there)."""
        return self.points[idx] * -player < 2

    def all_home(self, player):
        """True if every checker of 'player' still in play is in their home board."""
        if self.bar[player] > 0:
            return False
        outside = range(6, 24) if player == WHITE else range(0, 18)
        return all(self.count_on(i, player) == 0 for i in outside)

    def has_checkers_further(self, src, player):
        """
        True if 'player' has checkers in the home board further from bearing off than 'src'.
        Used for the rule that a higher die may bear off only from the furthest checker.
        """
        further = range(src + 1, 6) if player == WHITE else range(18, src)
        return any(self.count_on(i, player) > 0 for i in further)

    def winner(self):
        """Returns WHITE or BLACK if that player has borne off all 15 checkers, else None."""
        for p in (WHITE, BLACK):
            if self.off[p] == CHECKERS_PER_PLAYER:
                return p
        return None


def roll_dice():
    """Rolls two dice. Returns 4 copies of the value on a double (doubles are played 4 times)."""
    d1, d2 = random.randint(1, 6), random.randint(1, 6)
    return [d1] * 4 if d1 == d2 else [d1, d2]


def get_single_moves(state, player, die):
    """
    Returns every legal single-checker move for 'player' using one 'die' value.
    Each move is a tuple (src, dest): src is an index or 'bar', dest is an index or 'off'.
    Parameters:
      state: the GameState (board position) being examined
      player: WHITE (1) or BLACK (-1), whose checker or turn this is
      die: a single die value (1-6)
    """
    moves = []

    # Rule: a player with checkers on the bar must enter them before anything else
    if state.bar[player] > 0:
        dest = 24 - die if player == WHITE else die - 1
        if state.is_open(dest, player):
            moves.append(("bar", dest))
        return moves

    home_ready = state.all_home(player)
    direction = -1 if player == WHITE else 1

    for src in range(NUM_POINTS):
        if state.count_on(src, player) == 0:
            continue
        dest = src + direction * die
        if 0 <= dest < NUM_POINTS:
            if state.is_open(dest, player):
                moves.append((src, dest))
        elif home_ready:
            # Bearing off: exact roll, or a higher roll from the furthest checker
            distance = src + 1 if player == WHITE else 24 - src
            if die == distance or (die > distance and not state.has_checkers_further(src, player)):
                moves.append((src, "off"))
    return moves


def apply_move(state, player, move):
    """
    Returns a NEW state after 'player' makes 'move' (src, dest).
    Handles hitting a lone opponent checker (a 'blot') and bearing off.
    """
    new = state.copy()
    src, dest = move

    # Take the checker off its starting place
    if src == "bar":
        new.bar[player] -= 1
    else:
        new.points[src] -= player

    # Put it on its destination
    if dest == "off":
        new.off[player] += 1
    else:
        if new.points[dest] == -player:          # exactly one opponent checker -> hit it
            new.points[dest] = 0
            new.bar[-player] += 1
        new.points[dest] += player
    return new


def get_legal_turns(state, player, dice, dedupe=True):
    """
    Returns all legal complete turns for 'player' with the rolled 'dice'.
    Each result is (list_of_moves, resulting_state).
    dedupe : True keeps one turn per resulting position (faster for the AI);
             False keeps every move order (used by the GUI to validate clicks).
    Enforces: use as many dice as possible; if only one die can be used, use the larger one.
    """
    found = []  # entries: (moves, resulting_state, dice_used)

    def explore(current, remaining, moves_so_far, used):
        """
        Recursively tries every remaining die on every possible move and records each finished turn.
        Parameters:
          current: the position reached so far in this turn
          remaining: dice not yet used
          moves_so_far: moves already made this turn
          used: True if this die has already been played (drawn faded)
        """
        any_move = False
        tried = set()
        for i, die in enumerate(remaining):
            if die in tried:          # same die value gives same options, skip repeats
                continue
            tried.add(die)
            for mv in get_single_moves(current, player, die):
                any_move = True
                explore(apply_move(current, player, mv),
                        remaining[:i] + remaining[i + 1:],
                        moves_so_far + [mv], used + [die])
        if not any_move:
            found.append((moves_so_far, current, used))

    explore(state, list(dice), [], [])

    # Keep only turns that use the maximum number of dice
    max_len = max(len(m) for m, _, _ in found)
    turns = [t for t in found if len(t[0]) == max_len]

    # If only one die can be played and the dice differ, the larger die must be used if possible
    if max_len == 1 and len(dice) == 2 and dice[0] != dice[1]:
        larger = [t for t in turns if t[2][0] == max(dice)]
        if larger:
            turns = larger

    if not dedupe:
        return [(moves, result) for moves, result, _ in turns]

    # Remove turns that lead to exactly the same position
    unique = {}
    for moves, result, _ in turns:
        unique.setdefault(result.key(), (moves, result))
    return list(unique.values())


# ---------------------------------------------------------------------------
# Terminal display and test game (used for testing before the GUI is built)
# ---------------------------------------------------------------------------

def format_move(move):
    """Turns a move tuple into readable notation, e.g. (23, 17) -> '24/18'."""
    src, dest = move
    s = "bar" if src == "bar" else str(src + 1)
    d = "off" if dest == "off" else str(dest + 1)
    return f"{s}/{d}"


def print_board(state):
    """Prints a simple text version of the board.
    Parameters:
      state: the GameState (board position) being examined
    """
    def cell(i):
        """
        Returns a 4-character text cell for one point (e.g. "W3 ", "B5 " or " .  ").
        Parameters:
          i: board index of the point to format
        """
        n = state.points[i]
        return " .  " if n == 0 else (f"W{n:<3}" if n > 0 else f"B{-n:<3}")

    print("\n  " + "".join(f"{i:<4}" for i in range(13, 25)))
    print("  " + "".join(cell(i) for i in range(12, 24)))
    print("  " + "".join(cell(i) for i in range(11, -1, -1)))
    print("  " + "".join(f"{i:<4}" for i in range(12, 0, -1)))
    print(f"  Bar  W:{state.bar[WHITE]} B:{state.bar[BLACK]}   "
          f"Off  W:{state.off[WHITE]} B:{state.off[BLACK]}\n")


def random_choice(state, player, dice, turns):
    """Baseline 'Easy' strategy: picks any legal turn at random. Returns (moves, new_state).
    Parameters:
      state: the GameState (board position) being examined
      player: WHITE (1) or BLACK (-1), whose checker or turn this is
      dice: list of dice values still to play (4 copies on a double)
      turns: list of (moves, resulting_state) legal turns
    """
    return random.choice(turns)


def play_terminal_game(human=WHITE, computer=random_choice):
    """
    Runs a game in the terminal.
    human    : the side you control (WHITE/BLACK), or None to watch computer vs computer
    computer : function(state, player, dice, turns) -> (moves, new_state) used for the computer side
    """
    state = GameState()
    player = random.choice([WHITE, BLACK])

    while state.winner() is None:
        dice = roll_dice()
        turns = get_legal_turns(state, player, dice)
        if human is not None:
            print_board(state)
            print(f"{PLAYER_NAMES[player]} rolled {dice}")

        if not turns[0][0]:  # no legal moves at all
            if human is not None:
                print("No legal moves - turn passes.")
        elif player == human:
            for n, (moves, _) in enumerate(turns):
                print(f"  {n}: {' '.join(format_move(m) for m in moves)}")
            choice = -1
            while not 0 <= choice < len(turns):
                try:
                    choice = int(input("Choose a turn number: "))
                except ValueError:
                    pass
            state = turns[choice][1]
        else:
            moves, state = computer(state, player, dice, turns)
            if human is not None:
                print(f"Computer plays: {' '.join(format_move(m) for m in moves)}")
        player = -player

    if human is not None:
        print_board(state)
    print(f"{PLAYER_NAMES[state.winner()]} wins!")
    return state.winner()


if __name__ == "__main__":
    play_terminal_game(human=WHITE)
