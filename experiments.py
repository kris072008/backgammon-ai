"""
Backgammon with Expectiminimax AI - Experiments (Step 4)
Name:       Krishna Mahajan
Student ID: 106wucww
Module:     5FTC2067 Artificial Intelligence Principles - Mini Project

Plays silent AI-versus-AI games to compare the difficulty levels, tests how the beam width
changes the Hard AI's strength and speed, and compares the two search methods
(Expectiminimax vs Monte Carlo rollouts). Produces:
  - a results table printed to the terminal (and saved to results.csv)
  - results_winrate.png : win rate of the stronger AI in each match
  - results_beam.png    : Hard AI win rate and thinking time against beam width
  - results_expert.png  : the Expert (Monte Carlo) AI against Medium and Hard

Every game is seeded from 2026, its match number and its game number, so a run is exactly
reproducible, and the games can be shared between all the computer's cores without changing
any result.

Run:  python3 experiments.py          (full run, about 10-15 minutes on a 10-core laptop)
      python3 experiments.py --quick  (small version to check it works)
"""

import os
import sys
import csv
import math
import time
import random
from multiprocessing import Pool

from backgammon import WHITE, BLACK, GameState, roll_dice, get_legal_turns
from ai import easy_ai, medium_ai, hard_ai, expert_ai, expectiminimax_choice


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


def get_ai(name):
    """
    Returns the AI function for a name used in the experiments.
    Parameters:
      name: "Easy", "Medium", "Hard", "Expert", or "Hard-b<N>" for Hard with beam width N
    """
    if name.startswith("Hard-b"):
        return make_hard_ai(int(name[6:]))
    return {"Easy": easy_ai, "Medium": medium_ai, "Hard": hard_ai, "Expert": expert_ai}[name]


def play_seeded_game(job):
    """
    Plays one game from its own seed, so it gives the same result whichever core runs it.
    Parameters:
      job: (white AI name, black AI name, seed)
    Returns what play_game returns.
    """
    white, black, seed = job
    random.seed(seed)
    return play_game(get_ai(white), get_ai(black))


def run_match(name_a, name_b, games, match_no, pool):
    """
    Plays a match between two AIs, swapping colours halfway so neither side
    always gets the same colour. Games run in parallel on 'pool'.
    Parameters:
      name_a   : the first AI (its results are reported)
      name_b   : the second AI
      games    : number of games to play
      match_no : number of this match, used to give every game its own seed
      pool     : multiprocessing Pool that plays the games (None = play them one by one)
    Returns a dict with wins, win_rate (%), its 95% margin of error, gammons won by
    name_a and average thinking time per move (ms) for each side.
    """
    jobs, colours = [], []
    for g in range(games):
        a_is_white = g < games // 2
        jobs.append((name_a, name_b, 2026 * 1000000 + match_no * 10000 + g) if a_is_white
                    else (name_b, name_a, 2026 * 1000000 + match_no * 10000 + g))
        colours.append(WHITE if a_is_white else BLACK)
    results = pool.map(play_seeded_game, jobs) if pool else [play_seeded_game(j) for j in jobs]
    wins = gammons = 0
    time_a = time_b = 0.0
    moves_a = moves_b = 0
    for a_colour, (winner, gammon, t, m) in zip(colours, results):
        if winner == a_colour:
            wins += 1
            gammons += gammon
        time_a += t[a_colour]; moves_a += m[a_colour]
        time_b += t[-a_colour]; moves_b += m[-a_colour]
    rate = wins / games
    margin = 196 * math.sqrt(rate * (1 - rate) / games)       # 95% margin of error, in %
    return {"games": games, "wins": wins, "win_rate": 100 * rate, "margin": margin,
            "gammons": gammons, "ms_a": 1000 * time_a / max(moves_a, 1),
            "ms_b": 1000 * time_b / max(moves_b, 1)}


def make_hard_ai(beam):
    """
    Builds a Hard (depth-2 Expectiminimax) AI with a chosen beam width.
    Parameters:
      beam : number of candidate turns explored at each decision
    Returns an AI function with the usual (state, player, dice, turns) signature.
    """
    def ai(state, player, dice, turns):
        """Depth-2 Expectiminimax with the beam width chosen above.
        Parameters:
          state: the GameState before the AI's turn
          player: the side the AI plays
          dice: the dice the AI rolled
          turns: list of (moves, resulting_state) legal turns
        """
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


def save_expert_chart(expert_results):
    """
    Draws the Expert (Monte Carlo) AI's win rate against each opponent (skipped without matplotlib).
    Parameters:
      expert_results : list of (match name, result dict) from the method comparison
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = [n for n, _ in expert_results]
    rates = [r["win_rate"] for _, r in expert_results]
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=150)
    bars = ax.bar(names, rates, color="#8250be")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 1.5, f"{rate:.0f}%", ha="center")
    ax.axhline(50, ls="--", lw=1, color="#555")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Expert win rate (%)")
    ax.set_title("Monte Carlo rollouts (Expert) against Expectiminimax and greedy")
    fig.tight_layout()
    fig.savefig("results_expert.png")
    print("Chart saved: results_expert.png")


def main():
    """Runs the three experiments, prints the results table and saves the CSV and charts."""
    quick = "--quick" in sys.argv
    scale = 0.05 if quick else 1.0
    n = lambda full: max(2, int(full * scale))       # number of games to play in this mode
    pool = Pool(os.cpu_count())
    line = lambda name, r: (f"  {name:<20} {r['games']:>4} games  win rate {r['win_rate']:5.1f}% "
                            f"(+/- {r['margin']:4.1f})  gammons {r['gammons']:>3}  {r['ms_a']:6.1f} ms/move")

    print("Experiment 1: tournament between difficulty levels")
    tournament = []
    for no, (name, a, b, games) in enumerate([("Medium vs Easy", "Medium", "Easy", n(100)),
                                              ("Hard vs Easy", "Hard", "Easy", n(100)),
                                              ("Hard vs Medium", "Hard", "Medium", n(400))]):
        result = run_match(a, b, games, no, pool)
        tournament.append((name, result))
        print(line(name, result))

    print("\nExperiment 2: beam width of the Hard AI (vs Medium)")
    beam_results = []
    for no, beam in enumerate((1, 3, 6, 10), start=10):
        result = run_match(f"Hard-b{beam}", "Medium", n(200), no, pool)
        beam_results.append((beam, result))
        print(line(f"beam {beam}", result))

    print("\nExperiment 3: Monte Carlo rollouts (Expert) vs Expectiminimax (Hard) and greedy (Medium)")
    expert_results = []
    for no, (name, b, games) in enumerate([("Expert vs Medium", "Medium", n(300)),
                                           ("Expert vs Hard", "Hard", n(400))], start=20):
        result = run_match("Expert", b, games, no, pool)
        expert_results.append((name, result))
        print(line(name, result))
    pool.close()

    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "games", "wins", "win_rate_%", "margin_95_%", "gammons",
                         "ms_per_move_first", "ms_per_move_second"])
        for name, r in (tournament + [(f"Hard beam {b} vs Medium", r) for b, r in beam_results]
                        + expert_results):
            writer.writerow([name, r["games"], r["wins"], f"{r['win_rate']:.1f}", f"{r['margin']:.1f}",
                             r["gammons"], f"{r['ms_a']:.1f}", f"{r['ms_b']:.1f}"])
    print("\nResults saved: results.csv")
    save_charts(tournament, beam_results)
    save_expert_chart(expert_results)


if __name__ == "__main__":
    main()
