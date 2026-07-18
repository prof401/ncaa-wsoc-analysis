"""Tests for parse_scores_from_result."""

from __future__ import annotations

import unittest

from ncaa_wsoc.contests import ParsedResult, parse_scores_from_result


class ParseScoresFromResultTests(unittest.TestCase):
    def test_win_regulation(self) -> None:
        parsed = parse_scores_from_result("W 2-1")
        self.assertEqual(parsed, ParsedResult(goals_for=2, goals_against=1, ot_periods=None))

    def test_tie_regulation(self) -> None:
        parsed = parse_scores_from_result("T 0-0")
        self.assertEqual(parsed, ParsedResult(goals_for=0, goals_against=0, ot_periods=None))

    def test_loss_with_ot(self) -> None:
        parsed = parse_scores_from_result("L 0-1 (1 OT)")
        self.assertEqual(parsed, ParsedResult(goals_for=0, goals_against=1, ot_periods=1))

    def test_tie_with_multiple_ot(self) -> None:
        parsed = parse_scores_from_result("T 2-2 (3 OT)")
        self.assertEqual(parsed, ParsedResult(goals_for=2, goals_against=2, ot_periods=3))

    def test_case_insensitive(self) -> None:
        parsed = parse_scores_from_result("w 1-0")
        self.assertEqual(parsed, ParsedResult(goals_for=1, goals_against=0, ot_periods=None))

    def test_unparseable(self) -> None:
        self.assertIsNone(parse_scores_from_result(""))
        self.assertIsNone(parse_scores_from_result(None))
        self.assertIsNone(parse_scores_from_result("W"))
        self.assertIsNone(parse_scores_from_result("forfeit"))


if __name__ == "__main__":
    unittest.main()
