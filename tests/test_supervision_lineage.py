import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.zhengzhou.audit_supervision_lineage import (
    build_review_queue, join_memberships, summarize_memberships, verify_file,
)


class SupervisionLineageContracts(unittest.TestCase):
    def test_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            path.write_text("original", encoding="utf-8")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            inventory = {}
            verify_file(path, expected, inventory)
            path.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_file(path, expected, inventory)

    def test_conflicting_event_preserves_all_roles_and_folds(self):
        records = pd.DataFrame([
            {"event_candidate_id": "e", "native_cell_id": "c1", "split_role": "development", "group_id": "g1", "outer_fold": 0},
            {"event_candidate_id": "e", "native_cell_id": "c2", "split_role": "locked_internal_test", "group_id": "g2", "outer_fold": np.nan},
        ])
        result = summarize_memberships(records).iloc[0]
        self.assertTrue(result.presence_lineage_conflict)
        self.assertEqual(result.presence_role_unique, "MULTIPLE_OR_CONFLICTING_REVIEW_REQUIRED")
        self.assertTrue(result.has_development_presence_source)
        self.assertTrue(result.has_locked_test_presence_source)
        self.assertEqual(result.presence_native_cells_json, '["c1", "c2"]')

    def test_membership_season_cannot_be_inferred_from_shared_cell(self):
        links = pd.DataFrame([{"record_id": "r", "event_candidate_id": "e"}])
        ledger = pd.DataFrame([{"record_id": "r", "season_calendar": "winter"}])
        members = {s: pd.DataFrame(columns=["record_id", "native_cell_id"]) for s in ("spring", "summer", "autumn", "winter")}
        presences = {s: pd.DataFrame(columns=["native_cell_id", "split_role", "group_id", "outer_fold"]) for s in members}
        members["spring"] = pd.DataFrame([{"record_id": "r", "native_cell_id": "c"}])
        with self.assertRaisesRegex(ValueError, "Membership season differs"):
            join_memberships(links, ledger, members, presences)

    def test_unknown_original_record_cannot_receive_a_role(self):
        links = pd.DataFrame([{"record_id": "r", "event_candidate_id": "e"}])
        ledger = pd.DataFrame([{"record_id": "r", "season_calendar": "spring"}])
        members = {s: pd.DataFrame(columns=["record_id", "native_cell_id"]) for s in ("spring", "summer", "autumn", "winter")}
        presences = {s: pd.DataFrame(columns=["native_cell_id", "split_role", "group_id", "outer_fold"]) for s in members}
        members["spring"] = pd.DataFrame([{"record_id": "unknown", "native_cell_id": "c"}])
        with self.assertRaisesRegex(ValueError, "unknown original record"):
            join_memberships(links, ledger, members, presences)

    def test_blank_waterbird_summary_never_creates_absence_or_eligibility(self):
        common = {"source_file": "source", "source_sheet": "sheet", "source_record_count": 1,
                  "site_candidate_id": "site", "site_name_raw": "site", "start_time": "2025-01-01", "end_time": None,
                  "calendar_year": 2025, "season": "winter", "season_year": 2025, "source_crs": "UNCONFIRMED",
                  "crs_evidence": "pending", "event_id_status": "pending", "protocol": "unknown",
                  "complete_target_checklist": "unknown", "observed_waterbird_species_names_n": 0,
                  "taxonomy_unmatched_rows": 0, "possible_duplicate_rows": 0, "cross_event_collision_status": "pending"}
        events = pd.DataFrame([{**common, "event_candidate_id": "e", "source_kind": "monitoring"}])
        summary = pd.DataFrame(columns=["event_candidate_id", "presence_member_record_count", "presence_native_cells_json",
            "presence_roles_json", "presence_groups_json", "presence_outer_folds_json", "presence_role_unique",
            "presence_lineage_conflict", "has_development_presence_source", "has_locked_test_presence_source",
            "has_buffer_excluded_presence_source"])
        result = build_review_queue(events, summary).iloc[0]
        self.assertFalse(result.eligible_for_supervised_training)
        self.assertFalse(result.no_recorded_waterbirds_is_confirmed_non_detection)
        self.assertTrue(pd.isna(result.target_protection))
        self.assertTrue(pd.isna(result.target_restoration))
        self.assertTrue(result.historical_2025_already_viewed)
        self.assertEqual(result.source_crs, "UNCONFIRMED")
        self.assertEqual(result.source_independence_status, "UNKNOWN_SOURCE_INDEPENDENCE_REQUIRES_REVIEW")
        self.assertEqual(result.presence_role_unique, "NOT_APPLICABLE_NON_ND_BASE_MEMBERSHIP")

    def test_missing_nd_membership_is_distinct_from_non_nd_not_applicable(self):
        common = {"source_file": "source", "source_sheet": "sheet", "source_record_count": 1,
                  "site_candidate_id": "site", "site_name_raw": "site", "start_time": "2024-01-01", "end_time": None,
                  "calendar_year": 2024, "season": "winter", "season_year": 2024, "source_crs": "UNCONFIRMED",
                  "crs_evidence": "pending", "event_id_status": "pending", "protocol": "unknown",
                  "complete_target_checklist": "unknown", "observed_waterbird_species_names_n": 3,
                  "taxonomy_unmatched_rows": 0, "possible_duplicate_rows": 0, "cross_event_collision_status": "pending"}
        events = pd.DataFrame([{**common, "event_candidate_id": "nd_e", "source_kind": "nd"},
                               {**common, "event_candidate_id": "z_e", "source_kind": "zhuque"}])
        summary = pd.DataFrame(columns=["event_candidate_id", "presence_member_record_count", "presence_native_cells_json",
            "presence_roles_json", "presence_groups_json", "presence_outer_folds_json", "presence_role_unique",
            "presence_lineage_conflict", "has_development_presence_source", "has_locked_test_presence_source",
            "has_buffer_excluded_presence_source"])
        result = build_review_queue(events, summary).set_index("source_kind")
        self.assertEqual(result.loc["nd", "presence_role_unique"], "NO_POSITIVE_WATERBIRD_PRESENCE_MEMBERSHIP")
        self.assertEqual(result.loc["zhuque", "presence_role_unique"], "NOT_APPLICABLE_NON_ND_BASE_MEMBERSHIP")
        self.assertEqual(result.loc["zhuque", "observed_waterbird_species_names_n"], 3)
        self.assertFalse(result.eligible_for_supervised_training.any())


if __name__ == "__main__":
    unittest.main()
