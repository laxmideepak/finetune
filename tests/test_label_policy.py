"""Executable checks for the canonical label policy.

Run: python -m unittest discover -s tests -v
"""
import unittest

from src.label_policy import (
    LABELS, ONE_YEAR_DAYS, POLICY_TEXT, POLICY_VERSION,
    status_after_denial, status_from_cessation, validate_label,
)


class TestOneYearRule(unittest.TestCase):
    def test_four_months_is_current(self):
        self.assertEqual(status_from_cessation(120), ("CURRENT", None))

    def test_two_years_is_former(self):
        self.assertEqual(status_from_cessation(730), ("FORMER", None))

    def test_unspecified_cessation_is_unknown_timing(self):
        self.assertEqual(status_from_cessation(None), ("UNKNOWN", "timing-unresolved"))

    def test_imprecise_last_year_is_unknown(self):
        # "quit last year" cannot be resolved to either side of the boundary.
        self.assertEqual(status_from_cessation(None), ("UNKNOWN", "timing-unresolved"))

    def test_boundary_strictly_less_than_one_year(self):
        self.assertEqual(status_from_cessation(ONE_YEAR_DAYS - 1), ("CURRENT", None))
        self.assertEqual(status_from_cessation(ONE_YEAR_DAYS), ("FORMER", None))

    def test_exactly_one_year_is_former(self):
        self.assertEqual(status_from_cessation(365), ("FORMER", None))

    def test_former_label_with_recent_quit_is_current(self):
        # The explicit cessation date overrides the colloquial "former" word.
        self.assertEqual(status_from_cessation(90), ("CURRENT", None))

    def test_future_cessation_rejected(self):
        with self.assertRaises(ValueError):
            status_from_cessation(-1)

    def test_bool_is_not_a_number_of_days(self):
        with self.assertRaises(TypeError):
            status_from_cessation(True)

    def test_explicit_never_wins(self):
        self.assertEqual(status_from_cessation(30, explicit_lifetime_never=True), ("NEVER", None))


class TestDenialVsNever(unittest.TestCase):
    def test_denial_alone_is_not_never(self):
        self.assertEqual(status_after_denial(), ("UNKNOWN", "denial-only"))

    def test_explicit_lifetime_never(self):
        self.assertEqual(status_after_denial(explicit_lifetime_never=True), ("NEVER", None))


class TestContract(unittest.TestCase):
    def test_labels_exact(self):
        self.assertEqual(LABELS, ("CURRENT", "FORMER", "NEVER", "UNKNOWN"))

    def test_validate_label(self):
        self.assertEqual(validate_label(" former "), "FORMER")
        with self.assertRaises(ValueError):
            validate_label("SMOKER")

    def test_policy_text_contains_ratified_cases(self):
        for needle in ["four months", "two years", "exactly one year", "last year",
                       "unspecified", "three months", "denial-only", "one year",
                       "CURRENT", "FORMER", "NEVER", "UNKNOWN"]:
            self.assertIn(needle, POLICY_TEXT)

    def test_policy_version_present(self):
        self.assertEqual(POLICY_VERSION, "smoking-status-v1")


if __name__ == "__main__":
    unittest.main()
