"""
Backgammon with Expectiminimax AI - The AI (Step 2)
Name:       Krishna Mahajan
Student ID: 106wucww
Module:     5ENT1140 Artificial Intelligence Principles - Mini Project

How the AI thinks (Expectiminimax):
  - MAX node    : the AI's turn. It tries every legal turn and keeps the best value.
  - CHANCE node : the dice. It averages over all 21 possible rolls, weighted by probability.
  - MIN node    : the opponent's turn. It assumes the opponent picks the worst turn for the AI.
  - At the depth limit, positions are scored with evaluate().
"""

import random
import time
from backgammon import (WHITE, BLACK, PLAYER_NAMES, GameState, roll_dice,
                        get_legal_turns, random_choice, play_terminal_game)

# All 21 distinct dice rolls with their probabilities (doubles 1/36, others 2/36)
ALL_ROLLS = [((d1, d2), (1 / 36 if d1 == d2 else 2 / 36))
             for d1 in range(1, 7) for d2 in range(d1, 7)]

WIN_SCORE = 10000

# Weights for the evaluation function (tune these and report the effect!)
W_PIPS = 1.0         # each pip you are ahead in the race
W_BLOT = 4.0         # penalty for a blot the opponent can hit
W_SAFE_BLOT = 1.0    # smaller penalty for a blot out of reach
W_POINT = 2.0        # bonus for each point you own (2+ checkers)
W_HOME_POINT = 3.0   # extra bonus for points owned in your home board
W_BAR = 8.0          # each opponent checker on the bar (and penalty for yours)
W_OFF = 3.0          # each checker borne off


def pip_count(state, player):
    """Returns total pips 'player' still needs to move to bear everything off (lower = better).
    Parameters:
      state: the GameState (board position) being examined
      player: WHITE (1) or BLACK (-1), whose checker or turn this is
    """
    total = state.bar[player] * 25
    for i in range(24):
        n = state.count_on(i, player)
        if n:
            total += n * ((i + 1) if player == WHITE else (24 - i))
    return total


def is_exposed(state, idx, player):
    """True if an opponent checker (or the opponent's bar) is within 12 pips behind point 'idx'.
    Parameters:
      state: the GameState (board position) being examined
      idx: board index 0-23 of the point being checked
      player: WHITE (1) or BLACK (-1), whose checker or turn this is
    """
    opp = -player
    if state.bar[opp] > 0:
        entry_dist = (idx + 1) if player == WHITE else (24 - idx)
        if entry_dist <= 12:
            return True
    for j in range(24):
        if state.count_on(j, opp):
            dist = (idx - j) if player == WHITE else (j - idx)   # opponent moves toward idx
            if 0 < dist <= 12:
                return True
    return False


def evaluate(state, player):
    """
    Heuristic score of 'state' from 'player's point of view (higher = better for player).
    state  : the GameState to judge
    player : WHITE or BLACK, whose viewpoint the score is from
    """
    win = state.winner()
    if win is not None:
        return WIN_SCORE if win == player else -WIN_SCORE

    opp = -player
    score = W_PIPS * (pip_count(state, opp) - pip_count(state, player))
    score += W_BAR * (state.bar[opp] - state.bar[player])
    score += W_OFF * (state.off[player] - state.off[opp])

    home = range(0, 6) if player == WHITE else range(18, 24)
    for i in range(24):
        n = state.count_on(i, player)
        if n == 1:
            score -= W_BLOT if is_exposed(state, i, player) else W_SAFE_BLOT
        elif n >= 2:
            score += W_POINT + (W_HOME_POINT if i in home else 0)
    return score


def order_turns(turns, player, beam):
    """
    Sorts turns by their immediate evaluation and keeps only the best 'beam' of them.
    This 'beam search' keeps the tree small enough to run in a few seconds.
    Parameters:
      turns: list of (moves, resulting_state) legal turns
      player: WHITE (1) or BLACK (-1), whose checker or turn this is
      beam: number of candidate turns to keep (None = keep all)
    """
    ranked = sorted(turns, key=lambda t: evaluate(t[1], player), reverse=True)
    return ranked[:beam] if beam else ranked


def chance_node(state, mover, depth, ai_player, beam):
    """
    Averages the value over all 21 dice rolls for the player about to move ('mover').
    depth     : how many more turns to look ahead
    ai_player : the side the AI is playing (values are from its viewpoint)
    beam      : how many candidate turns to explore at each decision
    Parameters:
      state: the GameState (board position) being examined
      mover: the player about to roll and move at this node
      depth: number of turns still to look ahead
      ai_player: the side the AI plays; all values are from its viewpoint
      beam: number of candidate turns to keep (None = keep all)
    """
    if depth == 0 or state.winner() is not None:
        return evaluate(state, ai_player)
    expected = 0.0
    for (d1, d2), prob in ALL_ROLLS:
        dice = [d1] * 4 if d1 == d2 else [d1, d2]
        expected += prob * decision_node(state, mover, dice, depth, ai_player, beam)
    return expected


def decision_node(state, mover, dice, depth, ai_player, beam):
    """
    MAX node if 'mover' is the AI, MIN node if it is the opponent.
    Returns the best (or worst) value reachable with this roll.
    Parameters:
      state: the GameState (board position) being examined
      mover: the player about to roll and move at this node
      dice: list of dice values still to play (4 copies on a double)
      depth: number of turns still to look ahead
      ai_player: the side the AI plays; all values are from its viewpoint
      beam: number of candidate turns to keep (None = keep all)
    """
    turns = get_legal_turns(state, mover, dice)
    if not turns[0][0]:  # no legal move: turn passes
        return chance_node(state, -mover, depth - 1, ai_player, beam)

    turns = order_turns(turns, mover, beam)
    values = [chance_node(result, -mover, depth - 1, ai_player, beam) for _, result in turns]
    return max(values) if mover == ai_player else min(values)


def expectiminimax_choice(state, player, dice, turns, depth=2, beam=6):
    """
    Picks the AI's turn. Returns (moves, new_state).
    depth : 1 = judge only the position after our move (greedy)
            2 = also consider every opponent roll and their best reply
    beam  : number of candidate turns explored at each decision
    Parameters:
      state: the GameState (board position) being examined
      player: WHITE (1) or BLACK (-1), whose checker or turn this is
      dice: list of dice values still to play (4 copies on a double)
      turns: list of (moves, resulting_state) legal turns
      depth: number of turns still to look ahead
      beam: number of candidate turns to keep (None = keep all)
    """
    if not turns[0][0]:
        return turns[0]
    if len(turns) == 1:
        return turns[0]
    candidates = order_turns(turns, player, beam)
    best_value, best_turn = float("-inf"), candidates[0]
    for turn in candidates:
        value = chance_node(turn[1], -player, depth - 1, player, beam)
        if value > best_value:
            best_value, best_turn = value, turn
    return best_turn


# ---------------------------------------------------------------------------
# Difficulty levels
# ---------------------------------------------------------------------------

def easy_ai(state, player, dice, turns):
    """Easy: random legal moves.
    Parameters:
      state: the GameState (board position) being examined
      player: WHITE (1) or BLACK (-1), whose checker or turn this is
      dice: list of dice values still to play (4 copies on a double)
      turns: list of (moves, resulting_state) legal turns
    """
    return random_choice(state, player, dice, turns)


def medium_ai(state, player, dice, turns):
    """Medium: greedy, looks only at the position right after its own move (depth 1).
    Parameters:
      state: the GameState (board position) being examined
      player: WHITE (1) or BLACK (-1), whose checker or turn this is
      dice: list of dice values still to play (4 copies on a double)
      turns: list of (moves, resulting_state) legal turns
    """
    return expectiminimax_choice(state, player, dice, turns, depth=1, beam=None)


def hard_ai(state, player, dice, turns):
    """Hard: full Expectiminimax, considers all 21 opponent rolls and replies (depth 2).
    Parameters:
      state: the GameState (board position) being examined
      player: WHITE (1) or BLACK (-1), whose checker or turn this is
      dice: list of dice values still to play (4 copies on a double)
      turns: list of (moves, resulting_state) legal turns
    """
    return expectiminimax_choice(state, player, dice, turns, depth=2, beam=6)


# ---------------------------------------------------------------------------
# Experiments: AI vs AI (results go straight into your report)
# ---------------------------------------------------------------------------

def play_silent_game(white_ai, black_ai):
    """Plays one full game with no printing. Returns (winner, seconds used by each side).
    Parameters:
      white_ai: AI function controlling White
      black_ai: AI function controlling Black
    """
    state = GameState()
    player = random.choice([WHITE, BLACK])
    ais = {WHITE: white_ai, BLACK: black_ai}
    think_time = {WHITE: 0.0, BLACK: 0.0}
    moves_made = {WHITE: 0, BLACK: 0}
    while state.winner() is None:
        dice = roll_dice()
        turns = get_legal_turns(state, player, dice)
        start = time.time()
        _, state = ais[player](state, player, dice, turns)
        think_time[player] += time.time() - start
        moves_made[player] += 1
        player = -player
    return state.winner(), think_time, moves_made


def benchmark(ai_a, ai_b, name_a, name_b, games=20):
    """
    Plays 'games' games of ai_a vs ai_b (swapping colours halfway) and prints win rates
    and average thinking time per move.
    Parameters:
      ai_a: first AI function in the match
      ai_b: second AI function in the match
      name_a: display name for ai_a
      name_b: display name for ai_b
      games: number of games to play
    """
    wins = {name_a: 0, name_b: 0}
    time_used = {name_a: 0.0, name_b: 0.0}
    moves = {name_a: 0, name_b: 0}
    for g in range(games):
        a_is_white = g < games // 2
        white, black = (ai_a, ai_b) if a_is_white else (ai_b, ai_a)
        winner, t, m = play_silent_game(white, black)
        a_colour = WHITE if a_is_white else BLACK
        wins[name_a if winner == a_colour else name_b] += 1
        time_used[name_a] += t[a_colour]; moves[name_a] += m[a_colour]
        time_used[name_b] += t[-a_colour]; moves[name_b] += m[-a_colour]
        print(f"  game {g + 1}/{games}: {name_a if winner == a_colour else name_b} wins")
    print(f"\n{name_a} vs {name_b} over {games} games:")
    for n in (name_a, name_b):
        avg = 1000 * time_used[n] / max(moves[n], 1)
        print(f"  {n:<8} wins {wins[n]:>3} ({100 * wins[n] / games:.0f}%)   avg {avg:.1f} ms/move")


if __name__ == "__main__":
    print("1: Play against the Hard AI\n2: Benchmark (Medium vs Easy, Hard vs Medium)")
    if input("Choose: ").strip() == "2":
        benchmark(medium_ai, easy_ai, "Medium", "Easy", games=20)
        benchmark(hard_ai, medium_ai, "Hard", "Medium", games=10)
    else:
        play_terminal_game(human=WHITE, computer=hard_ai)
