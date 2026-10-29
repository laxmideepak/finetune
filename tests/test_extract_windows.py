"""Executable checks for the sentence-aligned extraction contract."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

from src.extract_windows import (
    DEMO_NOTE, WhitespaceOffsetTokenizer, extract, extract_windows, main,
)

TOKENIZER = WhitespaceOffsetTokenizer()
REPO_ROOT = Path(__file__).resolve().parents[1]


class TestSentenceAndSectionContract(unittest.TestCase):
    def test_sentence_alignment_and_section_metadata(self):
        note = (
            "Social History:\n"
            "Patient quit smoking four months ago. Denies current smoking.\n"
            "Family History:\n"
            "Mother smoked for 30 years.\n"
        )
        result = extract(note, TOKENIZER, radius=100)
        self.assertEqual(result["status"], "ok")
        headers = {section["header"] for section in result["sections"] if section["header"]}
        self.assertEqual(headers, {"SOCIAL HISTORY", "FAMILY HISTORY"})
        social = [crop for crop in result["crops"] if "SOCIAL HISTORY" in crop["section_headers"]]
        family = [crop for crop in result["crops"] if "FAMILY HISTORY" in crop["section_headers"]]
        self.assertTrue(social)
        self.assertTrue(family)
        self.assertTrue(all(len(crop["section_ids"]) == 1 for crop in result["crops"]))
        self.assertIn("four months ago", social[0]["text"])
        self.assertIn("Mother", family[0]["text"])

    def test_match_at_start_and_end(self):
        start = extract("Smoking: yes.", TOKENIZER)
        end = extract("The patient quit smoking", TOKENIZER)
        self.assertEqual(start["status"], "ok")
        self.assertEqual(end["status"], "ok")
        self.assertIn("Smoking", start["crops"][0]["text"])
        self.assertIn("smoking", end["crops"][0]["text"])

    def test_adjacent_triggers_merge(self):
        result = extract("Patient smokes cigarettes and cigars daily.", TOKENIZER)
        self.assertEqual(len(result["crops"]), 1)
        self.assertGreaterEqual(result["stats"]["n_triggers_used"], 2)

    def test_unicode_offsets_roundtrip(self):
        note = "Historia Social:\nJosé fumó tabaco durante años. La señora dejó el cigarro."
        result = extract(note, TOKENIZER)
        for crop in result["crops"]:
            self.assertEqual(note[crop["char_start"]:crop["char_end"]], crop["text"])
            self.assertEqual(hashlib.sha256(crop["text"].encode("utf-8")).hexdigest(), crop["text_sha256"])

    def test_no_matches_is_explicit(self):
        result = extract("The weather is pleasant today.", TOKENIZER)
        self.assertEqual(result["status"], "no-triggers")
        self.assertEqual(result["crops"], [])
        self.assertEqual(result["stats"]["n_crops"], 0)
        self.assertEqual(result["stats"]["n_triggers"], 0)

    def test_repeated_smoking_sections_are_kept(self):
        note = (
            "Social History:\nPatient smokes daily.\n"
            "Social History:\nPatient quit smoking two years ago.\n"
        )
        result = extract(note, TOKENIZER)
        section_ids = {section["section_id"] for section in result["sections"]}
        self.assertGreaterEqual(len(section_ids), 2)
        crop_section_ids = {crop["section_ids"][0] for crop in result["crops"]}
        self.assertGreaterEqual(len(crop_section_ids), 2)
        texts = " ".join(crop["text"] for crop in result["crops"])
        self.assertIn("smokes daily", texts)
        self.assertIn("quit smoking two years ago", texts)

    def test_cessation_clause_inside_sentence_survives_tiny_radius(self):
        note = (
            "Social History:\n"
            "The patient has a long history of tobacco use and after many years of smoking "
            "quit four months ago.\n"
        )
        result = extract(note, TOKENIZER, radius=1)
        self.assertEqual(result["status"], "ok")
        self.assertIn("four months ago", result["crops"][0]["text"])

    def test_validates_arguments(self):
        with self.assertRaises(ValueError):
            extract("Smoking.", TOKENIZER, radius=-1)
        with self.assertRaises(ValueError):
            extract("Smoking.", TOKENIZER, budget=0)
        with self.assertRaises(ValueError):
            extract("Smoking.", TOKENIZER, budget=100, wider_budget=50)
        with self.assertRaises(TypeError):
            extract(None, TOKENIZER)


class TestBudgetAndOverflow(unittest.TestCase):
    def test_over_budget_splits_and_routes_wider_context(self):
        note = "Social History:\n" + " ".join(
            f"Sentence {index} mentions smoking in detail and continues with filler words."
            for index in range(8)
        )
        result = extract(note, TOKENIZER, radius=100, budget=24, wider_budget=60)
        self.assertGreater(len(result["crops"]), 1)
        self.assertTrue(all(crop["n_tokens"] <= 24 for crop in result["crops"]))
        self.assertTrue(result["wider_context"])
        self.assertTrue(all(item["n_tokens"] <= 60 for item in result["wider_context"]))
        self.assertTrue(any(crop["overflow_of_budget"] for crop in result["crops"]))
        self.assertEqual(result["stats"]["tokens_omitted_from_crops"], 0)

    def test_single_sentence_over_crop_budget_is_truncated_and_routed(self):
        note = "Social History:\n" + " ".join(f"word{i}" for i in range(40)) + " smoking"
        result = extract(note, TOKENIZER, radius=100, budget=10, wider_budget=100)
        self.assertTrue(result["crops"])
        truncated = [crop for crop in result["crops"] if crop["truncated"]]
        self.assertTrue(truncated)
        self.assertTrue(all(crop["needs_manual_route"] for crop in truncated))
        self.assertGreater(result["stats"]["tokens_omitted_from_crops"], 0)
        self.assertTrue(result["wider_context"])
        self.assertGreaterEqual(sum(item["n_tokens"] for item in result["wider_context"]), 41)

    def test_evidence_over_1024_gets_multiple_explicit_wider_chunks(self):
        note = "Social History:\n" + " ".join(f"w{i}" for i in range(1200)) + " smoking"
        result = extract(note, TOKENIZER, radius=100, budget=100, wider_budget=100)
        self.assertEqual(result["status"], "ok")
        self.assertGreater(len(result["wider_context"]), 1)
        self.assertTrue(all(item["n_tokens"] <= 100 for item in result["wider_context"]))
        self.assertTrue(all(item["requires_multi_chunk_handling"] for item in result["wider_context"]))
        self.assertTrue(any(item["mid_sentence_split"] for item in result["wider_context"]))
        self.assertEqual(result["stats"]["wider_over_1024"], 0)
        self.assertEqual(result["stats"]["multi_chunk_wider_clusters"], 1)


class TestDedupeBoilerplateFamily(unittest.TestCase):
    def test_within_section_duplicate_first_wins(self):
        note = "Social History:\nPatient smokes daily.\nPatient smokes daily.\n"
        result = extract(note, TOKENIZER)
        self.assertEqual(result["stats"]["n_within_section_duplicates"], 1)
        self.assertEqual(len(result["crops"]), 1)
        aliased = [trigger for trigger in result["triggers"] if trigger["aliased_from_sent_id"] is not None]
        self.assertTrue(aliased)

    def test_boilerplate_never_seeds_a_crop(self):
        note = (
            "Social History:\n"
            "If you develop chest pain, call your doctor. "
            "Patient smokes one pack daily.\n"
        )
        result = extract(note, TOKENIZER)
        self.assertEqual(result["stats"]["n_boilerplate_sentences"], 1)
        self.assertTrue(result["crops"])
        self.assertNotIn("chest pain", result["crops"][0]["text"])
        self.assertIn("smokes one pack daily", result["crops"][0]["text"])

    def test_unpopulated_template_is_not_affirmative(self):
        note = "Social History:\nTobacco use: ___\n"
        result = extract(note, TOKENIZER)
        self.assertEqual(result["status"], "all-triggers-excluded")
        self.assertEqual(len(result["unpopulated_templates"]), 1)
        self.assertEqual(result["crops"], [])

    def test_family_margin_is_a_hint_not_a_label(self):
        note = "Family History:\nMother smoked for 30 years.\n"
        result = extract(note, TOKENIZER)
        self.assertTrue(result["crops"])
        self.assertTrue(any(trigger["family_margin"] for trigger in result["triggers"]))

    def test_crops_never_cross_section_boundaries(self):
        note = (
            "Social History:\nPatient smokes daily.\n"
            "Family History:\nMother smoked for 30 years.\n"
        )
        result = extract(note, TOKENIZER)
        for crop in result["crops"]:
            self.assertEqual(len(crop["section_ids"]), 1)


class TestCli(unittest.TestCase):
    def test_demo_cli(self):
        completed = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "extract_windows.py"), "--demo"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["crops"])

    def test_main_demo_in_process(self):
        self.assertEqual(main(["--demo"]), 0)


if __name__ == "__main__":
    unittest.main()
