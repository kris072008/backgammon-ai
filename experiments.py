"""
Backgammon with Expectiminimax AI - Experiments (Step 4)
Name:       Krishna Mahajan
Student ID: 106wucww
Module:     5ENT1140 Artificial Intelligence Principles - Mini Project

Plays silent AI-versus-AI games to compare the three difficulty levels, and tests
how the beam width changes the Hard AI's strength and speed. Produces:
  - a results table printed to the terminal (and saved to results.csv)
  - results_winrate.png : win rate of the stronger AI in each match
  - results_beam.png    : Hard AI win rate and thinking time against beam width

Run:  python3 experiments.py          (full run, a few minutes)
      python3 experiments.py --quick  (small version to check it works)
"""

import sys
import csv
import time
import random

from backgammon import WHITE, BLACK, GameState, roll_dice, get_legal_turns
from ai import easy_ai, medium_ai, hard_ai, expectiminimax_choice


def play_game(white_ai, black_ai):
    """
    Plays one complete game with no output.
    Parameters:
      white_ai: AI function controlling White, called as ai(state, player, dice, turns)
      black_ai: AI function controlling Black
    Returns (winner, gammon, think_time, moves_made):
      winner     : WHITE or BLACK
      gammon     : True if the loser bore off no checkers (a double win)
      think_time : dict player -> total seconds spent choosing moves
      moves_made : dict player -> number of turns played
    """
    state = GameState()
    player = random.choice([WHITE, BLACK])          # random first player, as in real play
    ais = {WHITE: white_ai, BLACK: black_ai}
    think_time = {WHITE: 0.0, BLACK: 0.0}
    moves_made = {WHITE: 0, BLACK: 0}
    while state.winner() is None:
        dice = roll_dice()
        turns = get_legal_turns(state, player, dice)
        start = time.perf_counter()
        _, state = ais[player](state, player, dice, turns)
        think_time[player] += time.perf_counter() - start
        moves_made[player] += 1
        player = -player
    winner = state.winner()
    gammon = state.off[-winner] == 0
    return winner, gammon, think_time, moves_made


def run_match(ai_a, ai_b, games):
    """
    Plays a match between two AIs, swapping colours halfway so neither side
    always gets the same colour.
    Parameters:
      ai_a  : the first AI function (its results are reported)
      ai_b  : the second AI function
      games : number of games to play
    Returns a dict with wins, win_rate (%), gammons won by ai_a and
    average thinking time per move (ms) for each side.
    """
    wins = gammons = 0
    time_a = time_b = 0.0
    moves_a = moves_b = 0
    for g in range(games):
        a_is_white = g < games // 2
        white, black = (ai_a, ai_b) if a_is_white else (ai_b, ai_a)
        a_colour = WHITE if a_is_white else BLACK
        winner, gammon, t, m = play_game(white, black)
        if winner == a_colour:
            wins += 1
            gammons += gammon
        time_a += t[a_colour]; moves_a += m[a_colour]
        time_b += t[-a_colour]; moves_b += m[-a_colour]
        print(f"    game {g + 1}/{games}", end="\r")
    print(" " * 30, end="\r")
    return {"games": games, "wins": wins, "win_rate": 100 * wins / games, "gammons": gammons,
            "ms_a": 1000 * time_a / max(moves_a, 1), "ms_b": 1000 * time_b / max(moves_b, 1)}


def make_hard_ai(beam):
    """
    Builds a Hard (depth-2 Expectiminimax) AI with a chosen beam width.
    Parameters:
      beam : number of candidate turns explored at each decision
    Returns an AI function with the usual (state, player, dice, turns) signature.
    """
    def ai(state, player, dice, turns):
        """Depth-2 Expectiminimax with the beam width chosen above (same parameters as hard_ai)."""
        return expectiminimax_choice(state, player, dice, turns, depth=2, beam=beam)
    return ai


def save_charts(tournament, beam_results):
    """
    Draws the two result charts with matplotlib (skipped if it is not installed).
    Parameters:
      tournament   : list of (match name, result dict) from the level tournament
      beam_results : list of (beam width, result dict) from the beam experiment
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: charts skipped (pip install matplotlib)")
        return

    names = [n for n, _ in tournament]
    rates = [r["win_rate"] for _, r in tournament]
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=150)
    bars = ax.bar(names, rates, color=["#d9a441", "#c0504d", "#c0504d"][:len(names)])
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 1.5, f"{rate:.0f}%", ha="center")
    ax.axhline(50, ls="--", lw=1, color="#555")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Win rate (%)")
    ax.set_title("Win rate of the stronger AI (first-named)")
    fig.tight_layout()
    fig.savefig("results_winrate.png")

    beams = [b for b, _ in beam_results]
    fig, ax1 = plt.subplots(figsize=(6, 3.6), dpi=150)
    ax1.plot(beams, [r["win_rate"] for _, r in beam_results], "o-", color="#c0504d", label="win rate vs Medium")
    ax1.set_xlabel("Beam width (turns explored per decision)")
    ax1.set_ylabel("Win rate vs Medium (%)", color="#c0504d")
    ax2 = ax1.twinx()
    ax2.plot(beams, [r["ms_a"] for _, r in beam_results], "s--", color="#3d7ea6", label="ms per move")
    ax2.set_ylabel("Thinking time (ms/move)", color="#3d7ea6")
    ax1.set_title("Hard AI: strength vs speed as beam width grows")
    fig.tight_layout()
    fig.savefig("results_beam.png")
    print("Charts saved: results_winrate.png, results_beam.png")


def main():
    """Runs both experiments, prints the results table and saves the CSV and charts."""
    quick = "--quick" in sys.argv
    random.seed(2026)                                # fixed seed so a run can be repeated
    scale = 0.1 if quick else 1.0
    n = lambda full: max(2, int(full * scale))       # number of games to play in this mode

    print("Experiment 1: tournament between difficulty levels")
    tournament = []
    for name, a, b, games in [("Medium vs Easy", medium_ai, easy_ai, n(100)),
                              ("Hard vs Easy", hard_ai, easy_ai, n(50)),
                              ("Hard vs Medium", hard_ai, medium_ai, n(100))]:
        result = run_match(a, b, games)
        tournament.append((name, result))
        print(f"  {name:<16} {result['games']:>4} games  win rate {result['win_rate']:5.1f}%  "
              f"gammons {result['gammons']:>3}  {result['ms_a']:6.1f} ms/move")

    print("\nExperiment 2: beam width of the Hard AI (vs Medium)")
    beam_results = []
    for beam in (1, 3, 6, 10):
        result = run_match(make_hard_ai(beam), medium_ai, n(40))
        beam_results.append((beam, result))
        print(f"  beam {beam:>2}   {result['games']:>4} games  win rate {result['win_rate']:5.1f}%  "
              f"{result['ms_a']:6.1f} ms/move")

    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "games", "wins", "win_rate_%", "gammons", "ms_per_move_first", "ms_per_move_second"])
        for name, r in tournament + [(f"Hard beam {b} vs Medium", r) for b, r in beam_results]:
            writer.writerow([name, r["games"], r["wins"], f"{r['win_rate']:.1f}", r["gammons"], f"{r['ms_a']:.1f}", f"{r['ms_b']:.1f}"])
    print("\nResults saved: results.csv")
    save_charts(tournament, beam_results)


if __name__ == "__main__":
    main()
