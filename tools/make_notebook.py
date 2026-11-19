"""
Backgammon with Expectiminimax AI - Builds the submission notebook
Name:       Krishna Mahajan
Student ID: 106wucww
Module:     5FTC2067 Artificial Intelligence Principles - Mini Project

The module asks for the code as a Jupyter notebook (.ipynb). This script puts every source file
into one notebook (each in a %%writefile cell, so running the notebook recreates the game),
adds cells that demonstrate the AI step by step, then executes it so the outputs are saved.
Run from the project folder:   python3 tools/make_notebook.py [output.ipynb]
"""

import os
import shutil
import sys
import tempfile

import nbformat
from nbclient import NotebookClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "Backgammon_AI_106wucww.ipynb")

MODULES = [
    ("backgammon.py", "Step 1 - Rules engine",
     "The board, the dice and every rule of backgammon: legal-turn generation (including the "
     "must-use-both-dice and larger-die rules), hitting, the bar and bearing off."),
    ("ai.py", "Step 2 - The AI",
     "The evaluation function, **Expectiminimax** (MAX, CHANCE and MIN nodes), **Monte Carlo "
     "rollouts**, the four difficulty levels, the calibrated win probability and the doubling "
     "cube decisions."),
    ("art.py", "Step 3 - Artwork drawn in code",
     "Every image in the game (wood, felt, points, checkers, dice, gold lettering) is drawn by "
     "these functions, so the game needs no image files."),
    ("gui.py", "Step 4 - Graphical interface",
     "The Pygame interface: menu, tutorial, hints and move grades, the search-tree view, the "
     "doubling cube, the game review and the AI-vs-AI watch mode."),
    ("main.py", "Step 5 - Entry point (the published game)",
     "Starts the game. The same file runs the desktop version and the browser version "
     "(served locally at http://localhost:8000 with pygbag)."),
    ("experiments.py", "Step 6 - Experiments",
     "The 2,100 seeded AI-vs-AI games behind the results, run in parallel on every core. "
     "Run it from a terminal (`python3 experiments.py`, about 10 minutes, or `--quick`)."),
    ("tests/test_backgammon_ai.py", "Step 7 - Automated tests",
     "19 tests of the rules, the scoring, the win probability, colour symmetry and every AI."),
]


def md(text):
    """Returns a markdown cell.
    Parameters:
      text: the markdown source
    """
    return nbformat.v4.new_markdown_cell(text.strip())


def code(text):
    """Returns a code cell.
    Parameters:
      text: the Python source
    """
    return nbformat.v4.new_code_cell(text.strip())


def build():
    """Assembles the notebook cells and returns the notebook (not yet executed)."""
    cells = [md("""
# Backgammon AI: Expectiminimax and Monte Carlo opponents

**Name:** Krishna Mahajan
**Student ID:** 106wucww
**Module:** 5FTC2067 Artificial Intelligence Principles, Mini Project
University of Hertfordshire (delivered at PSB Academy, Singapore)

This notebook contains **all of the code** for the project and **runs the AI live**:

1. The source files (each cell writes one file, so running the notebook rebuilds the game).
2. Demonstrations: the rules, the evaluation function, one Expectiminimax decision traced
   step by step, Monte Carlo rollouts, the win probability and doubling cube, a small live
   tournament and the automated tests.
3. The last cell **launches the published game** (a desktop window).

**To run:** `pip install pygame-ce matplotlib`, then *Run All*. Every function in the code has
a comment describing what it does and its parameters.
"""),
             code("""
# Check the two libraries the project needs
import importlib.util
for lib in ("pygame", "matplotlib"):
    print(f"{lib:11s}", "installed" if importlib.util.find_spec(lib) else "MISSING - run: pip install pygame-ce matplotlib")
"""),
             md("## Part 1: The source code")]
    for path, title, about in MODULES:
        with open(os.path.join(ROOT, path), encoding="utf8") as f:
            src = f.read().rstrip("\n")
        cells.append(md(f"### {title}: `{path}`\n\n{about}"))
        if "/" in path:
            cells.append(code(f"import os\nos.makedirs({os.path.dirname(path)!r}, exist_ok=True)"))
        cells.append(code(f"%%writefile {path}\n{src}"))

    cells += [
        md("""
## Part 2: The AI in action

### 2.1 The rules engine
The start position, and every legal way White can play an opening roll of 5-3.
"""),
        code("""
import importlib
import os
import random
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")      # no window is needed for the demos
sys.path.insert(0, os.getcwd())
backgammon = importlib.reload(importlib.import_module("backgammon"))   # the files written above
ai = importlib.reload(importlib.import_module("ai"))
WHITE, BLACK, GameState = backgammon.WHITE, backgammon.BLACK, backgammon.GameState
get_legal_turns, format_move = backgammon.get_legal_turns, backgammon.format_move
print_board = backgammon.print_board

start = GameState()
print_board(start)
turns = get_legal_turns(start, WHITE, [5, 3])
print(f"\\nWhite rolls 5-3: {len(turns)} different legal turns, for example:")
for moves, _ in turns[:6]:
    print("  ", " ".join(format_move(m) for m in moves))
"""),
        md("""
### 2.2 The evaluation function
How the AI scores a position (higher is better for the side named): pip-count lead, made
points, blots, checkers on the bar and checkers borne off. The start position is level.
"""),
        code("""
print("Pip counts at the start:", ai.pip_count(start, WHITE), "each side")
print("Score for White:", ai.evaluate(start, WHITE), "  Score for Black:", ai.evaluate(start, BLACK))
print("\\nThe six best 5-3 turns by immediate score (what the greedy Medium AI sees):")
for moves, result in ai.order_turns(turns, WHITE, 6):
    print(f"  {' '.join(format_move(m) for m in moves):14s} score {ai.evaluate(result, WHITE):6.1f}")
"""),
        md("""
### 2.3 One Expectiminimax decision, step by step
For each candidate turn (**MAX**), the AI averages over all 21 dice rolls the opponent could
throw, weighted 1/36 for doubles and 2/36 otherwise (**CHANCE**), and for each roll assumes the
opponent plays the reply that is worst for the AI (**MIN**, among the opponent's 6
best-looking replies: the beam that keeps the search fast). It picks the turn with the highest
expected value. The answer, 8/3 6/3, is the standard expert move for 5-3.
"""),
        code("""
def expected_value(result):
    \"\"\"Expectiminimax value of a turn: CHANCE over all 21 rolls, MIN reply for each.
    Parameters:
      result: the GameState after White's candidate turn
    \"\"\"
    return ai.chance_node(result, BLACK, 1, WHITE, 6)

print(f"{'Candidate turn':16s}{'Expected value':>15s}")
values = []
for moves, result in ai.order_turns(turns, WHITE, 6):
    values.append((expected_value(result), " ".join(format_move(m) for m in moves)))
for v, name in sorted(values, reverse=True):
    print(f"{name:16s}{v:15.2f}")

best_moves, _ = ai.hard_ai(start, WHITE, [5, 3], turns)
print("\\nHard AI (depth-2 Expectiminimax) plays:", " ".join(format_move(m) for m in best_moves))
"""),
        md("""
### 2.4 Monte Carlo rollouts (the Expert AI)
Instead of a fixed tree, the Expert tests its 6 best-looking turns by playing 24 short
simulated futures for each (4 turns, both sides greedy), using the **same dice** for every
candidate, and scores where each future ends with the calibrated win probability.
"""),
        code("""
random.seed(2026)
for rate, moves, _ in ai.rollout_values(start, WHITE, [5, 3], turns):
    print(f"  {' '.join(format_move(m) for m in moves):14s} average win chance {rate:6.1%}")
"""),
        md("""
### 2.5 Win probability and the doubling cube
A logistic curve over the two sides' evaluations, fitted on 60,672 positions, turns a score into
a win chance. The AI offers a double between 70% and 88% and accepts one at 25% or more.
"""),
        code("""
print(f"Start position, White to move: White wins {ai.win_probability(start, WHITE, WHITE):.1%}\\n")

def bear_off_race(white_off, black_far=False):
    \"\"\"Builds a race: both sides' remaining checkers spread over their home boards.
    Parameters:
      white_off: how many checkers White has already borne off (Black has none)
      black_far: True to put Black's checkers outside its home board (Black far behind in the race)
    \"\"\"
    s = GameState(points=[0] * 24, off={WHITE: white_off, BLACK: 0})
    for k in range(15 - white_off):
        s.points[k % 6] += 1                         # White's home board (points 1-6)
    for k in range(15):
        s.points[(12 if black_far else 18) + k % 6] -= 1
    return s

print(f"{'Race, White to move':34s}{'White wins':>11s}{'White doubles?':>16s}{'Black takes?':>14s}")
for label, s in (("White 3 checkers ahead", bear_off_race(3)),
                 ("White 6 checkers ahead", bear_off_race(6)),
                 ("Black far behind in the race", bear_off_race(0, black_far=True))):
    p = ai.win_probability(s, WHITE, WHITE)
    print(f"{label:34s}{p:11.1%}{str(ai.should_double(s, WHITE)):>16s}{str(ai.should_take(s, BLACK)):>14s}")
print("\\nAbove 88% the AI does not double: it is 'too good' and plays on for a gammon instead.")
"""),
        md("""
### 2.6 A small live tournament
A quick check that the stronger methods win (the full study of 2,100 games is in
`experiments.py` and the report). Seeded, so the result is the same every run.
"""),
        code("""
import time
def match(ai_a, ai_b, games, seed):
    \"\"\"Plays ai_a against ai_b, swapping colours each game. Returns ai_a's wins and the time taken.
    Parameters:
      ai_a: first AI function
      ai_b: second AI function
      games: number of games
      seed: random seed, so the match can be repeated exactly
    \"\"\"
    random.seed(seed)
    wins, t0 = 0, time.time()
    for g in range(games):
        white, black = (ai_a, ai_b) if g % 2 == 0 else (ai_b, ai_a)
        winner, _, _ = ai.play_silent_game(white, black)
        wins += (winner == WHITE) == (g % 2 == 0)
    return wins, time.time() - t0

for name, a, b, n in (("Medium vs Easy", ai.medium_ai, ai.easy_ai, 20),
                      ("Hard vs Medium", ai.hard_ai, ai.medium_ai, 10)):
    w, secs = match(a, b, n, seed=106)
    print(f"{name}: {w}/{n} wins for the first AI ({secs:.0f} s)")
"""),
        md("### 2.7 Results of the full study (2,100 seeded games, 95% margins of error)"),
        code("""
import matplotlib.pyplot as plt
names = ["Medium vs Easy", "Hard vs Easy", "Hard vs Medium", "Expert vs Medium", "Expert vs Hard"]
rates = [100, 99, 59, 72, 57]
errors = [0, 2, 5, 5, 5]
fig, axis = plt.subplots(figsize=(8, 3.6))
bars = axis.barh(names[::-1], rates[::-1], xerr=errors[::-1], color="#1f6b4f", capsize=4)
axis.axvline(50, color="#8a2a2a", linestyle="--", linewidth=1)
axis.set_xlim(0, 105)
axis.set_xlabel("Win rate of the first AI (%)")
axis.set_title("Look-ahead beats greedy play; Monte Carlo beats Expectiminimax")
for bar, r in zip(bars, rates[::-1]):
    axis.text(r + 1.5, bar.get_y() + bar.get_height() / 2, f"{r}%", va="center")
plt.tight_layout()
plt.show()
"""),
        md("### 2.8 Automated tests"),
        code("""
import unittest

os.environ["NO_COLOR"] = "1"                          # plain text output in the notebook
suite = unittest.defaultTestLoader.discover("tests")
result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
print("All tests passed" if result.wasSuccessful() else "Some tests failed")
"""),
        md("""
## Part 3: Play the game
Running this cell opens the game in its own window (it keeps running after the cell finishes).
Controls: **Space** roll, **H** hint, **T** search tree, click the cube or **D** to double,
**B** board set, **M** sound. On the title screen: **1-4** level, **W** watch two AIs,
**U** tutorial.
"""),
    ]
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def main():
    """Builds the notebook, runs it in a scratch folder so the outputs are saved, then adds the
    launch cell (left unrun, so building the notebook does not open a game window)."""
    nb = build()
    work = tempfile.mkdtemp(prefix="nb_")
    try:
        NotebookClient(nb, timeout=900, kernel_name="python3",
                       resources={"metadata": {"path": work}}).execute()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    nb.cells.append(code("""
import subprocess
import sys

game = subprocess.Popen([sys.executable, "main.py"])   # opens the game window
print("The game is starting in a new window.")
"""))
    nbformat.write(nb, OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
