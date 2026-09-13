"""Cached O*NET CSV access, rating cleanup, and the analytic-sample derivation.

The analytic sample (N = 894 in O*NET 30.3) is every occupation that has
non-suppressed IM and LV ratings in all four rated descriptor domains AND
task statements AND task-to-DWA mappings. This matches the derivation used
in the sibling transferability project, kept identical for comparability.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SOC = "O*NET-SOC Code"

# label -> (ratings CSV, ENABLES crosswalk CSV or None)
RATED_DOMAINS = {
    "EssentialSkill": ("essential_skills.csv", "essential_skills_to_work_activities.csv"),
    "TransferableSkill": ("transferable_skills.csv", "transferable_skills_to_work_activities.csv"),
    "Ability": ("abilities.csv", "abilities_to_work_activities.csv"),
    "Knowledge": ("knowledge.csv", None),
}

REQUIRED_FILES = [
    "occupation_data.csv", "task_statements.csv", "task_ratings.csv",
    "tasks_to_dwas.csv", "gwas_to_iwas_to_dwas.csv", "work_activities.csv",
    "essential_skills.csv", "transferable_skills.csv", "abilities.csv", "knowledge.csv",
    "essential_skills_to_work_activities.csv", "transferable_skills_to_work_activities.csv",
    "abilities_to_work_activities.csv", "work_styles_to_work_activities.csv",
    "content_model_reference.csv", "level_scale_anchors.csv", "scales_reference.csv",
    "related_occupations.csv", "job_zones.csv", "job_titles.csv",
    "sample_of_reported_titles.csv", "software_skills.csv",
]


class OnetData:
    """Read-only, memoized access to the O*NET CSV folder."""

    def __init__(self, data_dir: str | Path):
        self.dir = Path(data_dir)
        if not self.dir.is_dir():
            raise FileNotFoundError(
                f"O*NET data folder not found: {self.dir}. "
                "Set ONET_DATA_DIR in .env or run pipeline/00_check_data.py --download."
            )
        self._cache: dict[str, pd.DataFrame] = {}

    def load(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            self._cache[name] = pd.read_csv(self.dir / name)
        return self._cache[name]

    def clean_ratings(self, fn: str) -> tuple[pd.DataFrame, int]:
        """Drop suppressed rows, pivot IM/LV wide, flag Not Relevant.

        Returns (frame with columns [SOC, Element ID, importance, level,
        not_relevant], number of suppressed rows dropped).
        """
        d = self.load(fn)
        n_suppressed = int((d["Recommend Suppress"] == "Y").sum())
        d = d[d["Recommend Suppress"] != "Y"]
        im = d[d["Scale ID"] == "IM"][[SOC, "Element ID", "Data Value"]].rename(
            columns={"Data Value": "importance"})
        lv = d[d["Scale ID"] == "LV"][[SOC, "Element ID", "Data Value", "Not Relevant"]].rename(
            columns={"Data Value": "level"})
        lv = lv.copy()
        lv["not_relevant"] = lv["Not Relevant"] == "Y"
        lv.loc[lv["not_relevant"], "level"] = 0.0
        m = im.merge(lv[[SOC, "Element ID", "level", "not_relevant"]],
                     on=[SOC, "Element ID"], how="outer")
        m["not_relevant"] = m["not_relevant"].eq(True)
        return m, n_suppressed

    def task_importance(self) -> pd.DataFrame:
        """Mean IM rating per (SOC, Task ID) from task_ratings, suppression-filtered."""
        d = self.load("task_ratings.csv")
        d = d[(d["Scale ID"] == "IM") & (d["Recommend Suppress"] != "Y")]
        return (d.groupby([SOC, "Task ID"])["Data Value"].mean()
                .rename("importance").reset_index())

    def analytic_sample(self) -> list[str]:
        jobs = None
        for fn, _ in RATED_DOMAINS.values():
            d = self.load(fn)
            d = d[d["Recommend Suppress"] != "Y"]
            j = set(d[d["Scale ID"] == "IM"][SOC]) & set(d[d["Scale ID"] == "LV"][SOC])
            jobs = j if jobs is None else jobs & j
        ts = self.load("task_statements.csv")
        td = self.load("tasks_to_dwas.csv")
        return sorted(jobs & set(ts[SOC]) & set(td[SOC]))

    def alt_titles(self) -> pd.Series:
        """SOC -> sorted list of unique alternate/reported titles (for entity linking)."""
        jt = self.load("job_titles.csv")[[SOC, "Job Title"]].rename(columns={"Job Title": "t"})
        rt = self.load("sample_of_reported_titles.csv")[[SOC, "Reported Job Title"]].rename(
            columns={"Reported Job Title": "t"})
        both = pd.concat([jt, rt], ignore_index=True).dropna(subset=["t"])
        return both.groupby(SOC)["t"].agg(lambda s: sorted(set(s)))
