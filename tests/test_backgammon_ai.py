"""
Backgammon with Expectiminimax AI - Automated tests (Step 6)
Name:       Krishna Mahajan
Student ID: 106wucww
Module:     5FTC2067 Artificial Intelligence Principles - Mini Project

Checks the rules engine, the scoring, the win probability and the AIs against known answers.
Run from the project folder:   python3 -m unittest discover tests
(pytest also runs these tests unchanged:  pytest tests)
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")      # gui.py imports pygame; no window is opened

from backgammon import (WHITE, BLACK, GameState, get_legal_turns, get_single_moves, apply_move,
                        roll_dice, format_move)
import ai
from gui import game_points


def position(white, black, bar=None):
    """
    Builds a test position from the checkers each side has on each point.
    Parameters:
      white: dict board index -> number of White checkers
      black: dict board index -> number of Black checkers
      bar: optional dict player -> checkers on the bar
    """
    points = [0] * 24
    for idx, n in white.items():
        points[idx] = n
    for idx, n in black.items():
        points[idx] = -n
    bar = bar or {WHITE: 0, BLACK: 0}
    off = {WHITE: 15 - sum(white.values()) - bar[WHITE], BLACK: 15 - sum(black.values()) - bar[BLACK]}
    return GameState(points, dict(bar), off)


def mirror(state):
    """
    Returns the same position with the colours swapped (White's checkers become Black's on
    the mirrored point, and the other way round).
    Parameters:
      state: the GameState to mirror
    """
    return GameState([-state.points[23 - i] for i in range(24)],
                     {WHITE: state.bar[BLACK], BLACK: state.bar[WHITE]},
                     {WHITE: state.off[BLACK], BLACK: state.off[WHITE]})


def turn_text(moves):
    """
    Writes a turn in board notation, e.g. '8/3 6/3'.
    Parameters:
      moves: list of (src, dest) moves
    """
    return " ".join(format_move(m) for m in moves)


class RulesTests(unittest.TestCase):
    """The rules engine: start position, legal turns, hitting, the bar and bearing off."""

    def test_start_position(self):
        """Each side starts with 15 checkers and a pip count of 167."""
        s = GameState()
        for p in (WHITE, BLACK):
            self.assertEqual(sum(s.count_on(i, p) for i in range(24)), 15)
            self.assertEqual(ai.pip_count(s, p), 167)

    def test_opening_3_1_has_16_turns(self):
        """An opening 3-1 has 16 different legal turns (the number quoted in the report)."""
        self.assertEqual(len(get_legal_turns(GameState(), WHITE, [3, 1])), 16)

    def test_doubles_are_played_four_times(self):
        """A double moves four times when there is room to."""
        turns = get_legal_turns(GameState(), WHITE, [6, 6, 6, 6])
        self.assertTrue(all(len(moves) == 4 for moves, _ in turns))

    def test_larger_die_rule(self):
        """If only one die can be played, it must be the larger one when that is possible."""
        # White's checker on 14 can play the 6 (14/8) or the 5 (14/9), but then the other die
        # is blocked, so only one die can be used and it must be the 6
        s = position({0: 9, 1: 1, 3: 4, 13: 1},
                     {2: 2, 4: 1, 5: 1, 8: 1, 11: 2, 12: 1, 21: 1, 22: 1, 23: 5})
        self.assertEqual([format_move(m) for m in get_single_moves(s, WHITE, 5)], ["14/9"])
        turns = get_legal_turns(s, WHITE, [5, 6])
        self.assertEqual([turn_text(m) for m, _ in turns], ["14/8"])

    def test_hitting_a_blot_sends_it_to_the_bar(self):
        """Landing on a single opposing checker puts it on the bar."""
        s = position({12: 1}, {9: 1})
        after = apply_move(s, WHITE, (12, 9))
        self.assertEqual(after.bar[BLACK], 1)
        self.assertEqual(after.points[9], 1)

    def test_bar_checker_must_enter_first(self):
        """With a checker on the bar, the only legal moves are entering moves."""
        s = position({5: 5, 7: 3, 12: 5, 23: 1}, {0: 2, 11: 5, 16: 3, 18: 5}, {WHITE: 1, BLACK: 0})
        for die in range(1, 7):
            self.assertTrue(all(src == "bar" for src, _ in get_single_moves(s, WHITE, die)))

    def test_blocked_entry_passes_the_turn(self):
        """If every entry point is blocked, the player cannot move at all."""
        s = position({5: 14}, {18: 3, 19: 2, 20: 2, 21: 2, 22: 3, 23: 3}, {WHITE: 1, BLACK: 0})
        self.assertEqual(get_legal_turns(s, WHITE, [4, 2]), [([], s)])

    def test_bearing_off_exact_and_higher_die(self):
        """A 6 bears off from point 6; a 6 bears off from the highest point when 6 is empty."""
        s = position({5: 1, 1: 1}, {23: 15})
        self.assertIn((5, "off"), get_single_moves(s, WHITE, 6))
        s2 = position({3: 1, 1: 1}, {23: 15})
        self.assertIn((3, "off"), get_single_moves(s2, WHITE, 6))
        self.assertNotIn((1, "off"), get_single_moves(s2, WHITE, 6))   # not while point 4 is occupied

    def test_moves_are_colour_symmetric(self):
        """Swapping the colours of a position gives exactly the mirrored legal turns, in the
        same order (the property whose absence caused the colour bias)."""
        random.seed(11)
        for _ in range(20):
            s, p = GameState(), WHITE
            for _ in range(random.randint(0, 25)):
                if s.winner() is not None:
                    break
                _, s = random.choice(get_legal_turns(s, p, roll_dice()))
                p = -p
            dice = roll_dice()
            mine = [mirror(r).key() for _, r in get_legal_turns(s, p, dice)]
            theirs = [r.key() for _, r in get_legal_turns(mirror(s), -p, dice)]
            self.assertEqual(mine, theirs)


class ScoringTests(unittest.TestCase):
    """Points at the end of a game: single, gammon and backgammon, times the cube."""

    def test_single_game(self):
        """The loser bore off a checker: one point times the cube."""
        s = position({}, {23: 14})
        self.assertEqual(game_points(s, WHITE, 2), (2, "single"))

    def test_gammon(self):
        """The loser bore off nothing: two points times the cube."""
        s = position({}, {20: 15})
        self.assertEqual(game_points(s, WHITE, 1), (2, "gammon"))

    def test_backgammon(self):
        """A gammon with a loser's checker in the winner's home board: three points."""
        s = position({}, {20: 14, 2: 1})
        self.assertEqual(game_points(s, WHITE, 4), (12, "backgammon"))


class AITests(unittest.TestCase):
    """The evaluation, the win probability, the cube decisions and the four AIs."""

    def test_hard_ai_finds_the_expert_opening(self):
        """On the opening 5-3 the Hard AI plays 8/3 6/3, with the expected value in the report."""
        s = GameState()
        turns = get_legal_turns(s, WHITE, [5, 3])
        moves, result = ai.hard_ai(s, WHITE, [5, 3], turns)
        self.assertEqual(turn_text(moves), "8/3 6/3")
        self.assertAlmostEqual(ai.chance_node(result, BLACK, 1, WHITE, 6), 15.83, places=2)

    def test_evaluation_of_a_finished_game(self):
        """A won game scores +WIN_SCORE for the winner and -WIN_SCORE for the loser."""
        s = position({}, {20: 15})
        self.assertEqual(ai.evaluate(s, WHITE), ai.WIN_SCORE)
        self.assertEqual(ai.evaluate(s, BLACK), -ai.WIN_SCORE)

    def test_win_probability_is_colour_symmetric(self):
        """The start position gives the side about to roll the same chance, whichever colour it is."""
        s = GameState()
        self.assertAlmostEqual(ai.win_probability(s, WHITE, WHITE), ai.win_probability(s, BLACK, BLACK))
        self.assertAlmostEqual(ai.win_probability(s, WHITE, WHITE) + ai.win_probability(s, BLACK, WHITE), 1.0)

    def test_cube_thresholds(self):
        """The AI takes a double at 25% or more and offers one only between 70% and 88%."""
        self.assertEqual(ai.TAKE_POINT, 0.25)
        self.assertEqual(ai.DOUBLE_WINDOW, (0.70, 0.88))
        self.assertFalse(ai.should_double(GameState(), WHITE))       # an even game: no double

    def test_every_ai_returns_a_legal_turn(self):
        """Easy, Medium, Hard and Expert always choose one of the legal turns they were given."""
        random.seed(5)
        s, p = GameState(), WHITE
        for _ in range(12):
            dice = roll_dice()
            turns = get_legal_turns(s, p, dice)
            keys = {r.key() for _, r in turns}
            for choose in (ai.easy_ai, ai.medium_ai, ai.hard_ai, ai.expert_ai):
                _, result = choose(s, p, dice, turns)
                self.assertIn(result.key(), keys)
            _, s = ai.medium_ai(s, p, dice, turns)
            p = -p

    def test_expert_is_repeatable_with_a_seed(self):
        """With the same random seed, the Expert's Monte Carlo search picks the same turn."""
        s = GameState()
        turns = get_legal_turns(s, WHITE, [6, 1])
        random.seed(3)
        first = ai.expert_ai(s, WHITE, [6, 1], turns)[0]
        random.seed(3)
        self.assertEqual(ai.expert_ai(s, WHITE, [6, 1], turns)[0], first)


class ExperimentTests(unittest.TestCase):
    """The experiments are reproducible: the same seeds give the same results."""

    def test_match_is_reproducible(self):
        """Running the same small match twice gives identical wins and gammons."""
        import experiments
        a = experiments.run_match("Medium", "Easy", 4, 99, None)
        b = experiments.run_match("Medium", "Easy", 4, 99, None)
        self.assertEqual((a["wins"], a["gammons"]), (b["wins"], b["gammons"]))


if __name__ == "__main__":
    unittest.main()
