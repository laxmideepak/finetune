from datetime import date

import pytest

from src.quit_resolver import apply_one_year_rule, resolve_quit

NOTE = date(2025, 6, 1)


def label(text, status="FORMER", age=60):
    return apply_one_year_rule(status, resolve_quit(text, note_date=NOTE, age=age), note_date=NOTE)[0]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("He quit smoking four months ago.", "CURRENT"),  # policy canonical: < 1 year
        ("She quit smoking two years ago.", "FORMER"),
        ("Former smoker, quit 2010.", "FORMER"),
        ("Former smoker, cessation date unspecified.", "UNKNOWN"),  # timing-unresolved
        ("Quit date: 12/12/2024", "CURRENT"),
        ("Quit date: 5/1/2024", "FORMER"),
        ("quit 2007, relapsed 2013, quit again 2019", "FORMER"),
        ("Ex-smoker - Quit x 6 yrs.", "FORMER"),
        ("smoke-free for 34 years", "FORMER"),
        ("Smoked 1 ppd x 30 yrs, quit Oct 2024.", "CURRENT"),
        ("Quit drinking in 2016. Former smoker.", "UNKNOWN"),  # alcohol date is not a smoking date
        ("Former smoker seen today, 06/01/2025.", "UNKNOWN"),  # visit date is not a quit date
        ("Plans to quit on 7/1/2025.", "UNKNOWN"),  # future/planned date
        ("quit smoking 4 years after her heart attack, which happened in 2013", "FORMER"),
        ("quit when her granddaughter was born; she is now 3 years old", "FORMER"),
    ],
)
def test_one_year_rule(text, expected):
    assert label(text) == expected


def test_quit_at_age_boundary_is_unresolved():
    # Quit at age 59 in a 60-year-old: anywhere from 0 to 2 years ago.
    assert label("stopped smoking at age 59", age=60) == "UNKNOWN"
    assert label("stopped smoking at age 50", age=60) == "FORMER"


def test_non_former_status_passes_through():
    assert label("quit 3 months ago", status="CURRENT") == "CURRENT"
    assert label("never smoked", status="NEVER") == "NEVER"
