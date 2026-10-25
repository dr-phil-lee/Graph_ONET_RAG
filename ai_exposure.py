"""Opportunity Data Occupation AI Exposure Index, joined onto O*NET occupations.

Source: Rowe, B. (2026). Opportunity Data AI Exposure Index (methodology v2,
O*NET 31.0 / BLS OEWS May 2025). https://opportunitydata.org/ai-exposure-methodology.html
The occupation table (https://opportunitydata.org/ai-exposure-occupations.html)
is offered as a CSV "for research and institutional use"; data/ai_exposure/
holds that table (696 SOC occupations).

How a score is built (from the methodology page, reproduced by METHOD_* below):
  1. Nine O*NET 31.0 descriptors (5 Work Activities, 4 Work Context), raw
     1-5 ratings, min-max normalized to 0-1 across the 911 O*NET occupations
     that carry all nine; "Spend Time Sitting" is inverted.
  2. Three dimensions = simple means of three normalized variables each:
     Digital (raises exposure), Human contact and Physical (protect).
  3. Composite = (Digital - Contact - Physical + 2) / 3, on 0-1
     (0 = fully protected, 1 = fully exposed); equal weights 1:1:1.
Every row of the stored table satisfies step 3 to 4 decimals.

The chatbot reports scores only. The source site also groups scores into
named bands; those labels are deliberately not used here.

Joining rules (kept strict so nothing is attributed wrongly):
  - The index is keyed by 6-digit SOC 2018 codes ("41-9041"); O*NET codes
    add a 2-digit detail suffix ("41-9041.00"). Each index row is scored
    from ONE O*NET occupation, so a row is joined only when its title equals
    the O*NET title (case and punctuation ignored). That covers every ".00"
    code in the index and the 8 rows scored from a detailed code (e.g.
    15-2099 "Bioinformatics Technicians" = O*NET 15-2099.01). A detailed
    O*NET code never inherits the score of a differently titled row.
  - Occupations missing from the index (e.g. Chief Executives) get no score;
    nothing is imputed.
  - The index uses O*NET 31.0 while the chatbot graph is O*NET 30.3; every
    record states this.
"""
from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import bootstrap

INDEX_CSV = bootstrap.DATA_DIR / "ai_exposure" / "opportunity_data_ai_exposure_index.csv"

SOURCE_NAME = "Opportunity Data Occupation AI Exposure Index"
SOURCE_CITATION = ("Rowe, B. (2026). Opportunity Data Occupation AI Exposure Index. "
                   "Opportunity Data. opportunitydata.org/ai-exposure-occupations")
SOURCE_URL = "https://opportunitydata.org/ai-exposure-occupations.html"
METHOD_URL = "https://opportunitydata.org/ai-exposure-methodology.html"
SOURCE_NOTE = ("Third-party index (not part of O*NET), scored from O*NET 31.0 work "
               "activity and work context ratings; employment = BLS OEWS May 2025 "
               "national estimates.")
FORMULA = "AI exposure = (Digital - Human contact - Physical + 2) / 3"

# (variable, O*NET element id, O*NET section and scale, dimension, direction)
VARIABLES: list[tuple[str, str, str, str, str]] = [
    ("Working with Computers", "4.A.3.b.1", "Work Activities, importance 1-5",
     "Digital", "higher = more exposed"),
    ("Analyzing Data or Information", "4.A.2.a.4", "Work Activities, importance 1-5",
     "Digital", "higher = more exposed"),
    ("Processing Information", "4.A.2.a.2", "Work Activities, importance 1-5",
     "Digital", "higher = more exposed"),
    ("Assisting and Caring for Others", "4.A.4.a.5", "Work Activities, importance 1-5",
     "Human contact", "higher = more protected"),
    ("Performing for or Working Directly with the Public", "4.A.4.a.8",
     "Work Activities, importance 1-5", "Human contact", "higher = more protected"),
    ("Face-to-Face Discussions with Individuals and Within Teams", "4.C.1.a.2.l",
     "Work Context, 1-5", "Human contact", "higher = more protected"),
    ("Spend Time Using Your Hands", "4.C.2.d.1.g", "Work Context, 1-5",
     "Physical", "higher = more protected"),
    ("Spend Time Sitting (inverted)", "4.C.2.d.1.a", "Work Context, 1-5",
     "Physical", "more sitting = less protected"),
    ("Responsible for Others' Health and Safety", "4.C.1.c.1", "Work Context, 1-5",
     "Physical", "higher = more protected"),
]

# Plain-language meaning of the score, shown before any calculation detail.
MEANING = ("The AI exposure score shows how much of an occupation's day-to-day work "
           "is exposed to AI-driven task automation. 0 means the work is largely "
           "shielded from it (hands-on, physical or face-to-face work); 1 means most "
           "of the work is the kind of digital, computer-based activity that today's AI "
           "can do or strongly assist. It measures exposure of tasks, not whether the "
           "job will disappear or how fast employers will adopt AI.")

METHOD_STEPS: list[str] = [
    "Take nine O*NET 31.0 ratings (1-5) for each occupation: 5 Work Activities "
    "and 4 Work Context items, as reported by incumbent workers and analysts.",
    "Min-max normalize each rating to 0-1 across the 911 O*NET occupations that "
    "have all nine; invert \"Spend Time Sitting\" so more sitting means less protection.",
    "Average three variables per dimension: Digital work intensity (raises "
    "exposure), Human contact intensity and Physical task intensity (both protect).",
    "Combine with equal weights: (Digital - Human contact - Physical + 2) / 3, "
    "giving 0 = fully protected and 1 = fully exposed.",
]

LIMITATIONS: list[str] = [
    "Measures task overlap with current AI capability; it does not predict job loss "
    "or adoption speed.",
    "Built on self-reported O*NET survey ratings; one score per SOC code.",
    "The 1:1:1 weighting is a modeling choice, not an empirical finding.",
]

REFERENCES: list[dict[str, str]] = [
    {"label": "Methodology",
     "citation": "Rowe, B. (2026). Opportunity Data AI Exposure Index. Opportunity "
                 "Data. Methodology page (methodology v2; released April 2026, "
                 "updated October 2026).",
     "url": METHOD_URL},
    {"label": "Occupation scores",
     "citation": "Rowe, B. (2026). Opportunity Data Occupation AI Exposure Index. "
                 "Opportunity Data. Occupation table (696 SOC occupations), "
                 "provided as CSV for research and institutional use.",
     "url": SOURCE_URL},
    {"label": "Input ratings",
     "citation": "National Center for O*NET Development. O*NET 31.0 Database "
                 "(August 2026), Work Activities and Work Context. U.S. Department "
                 "of Labor, Employment and Training Administration.",
     "url": "https://www.onetcenter.org/database.html"},
    {"label": "Employment",
     "citation": "U.S. Bureau of Labor Statistics. Occupational Employment and Wage "
                 "Statistics (OEWS), May 2025 national estimates.",
     "url": "https://www.bls.gov/oes/"},
    {"label": "Validation",
     "citation": "Opportunity Data. AI Exposure Index vs. BLS AI Exposure Categories "
                 "(August 2026): rank correlation 0.76 across 761 occupations.",
     "url": "https://opportunitydata.org/ai-exposure-bls-comparison.html"},
]

# continuous color scale for scores (green = protected .. red = exposed);
# no named bands
COLOR_STOPS: list[tuple[float, str]] = [(0.25, "#2E7D5B"), (0.55, "#C9A227"),
                                        (0.85, "#B23A3A")]


def _rgb(c: str) -> tuple[int, int, int]:
    return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)


def score_color(score: float) -> str:
    """Hex color for a composite score, interpolated along COLOR_STOPS."""
    s = float(score)
    if s <= COLOR_STOPS[0][0]:
        return COLOR_STOPS[0][1]
    for (a, ca), (b, cb) in zip(COLOR_STOPS, COLOR_STOPS[1:]):
        if s <= b:
            t = (s - a) / (b - a)
            rgb = [round(x + (y - x) * t) for x, y in zip(_rgb(ca), _rgb(cb))]
            return "#" + "".join(f"{v:02X}" for v in rgb)
    return COLOR_STOPS[-1][1]


@dataclass(frozen=True)
class Exposure:
    onet_soc: str            # O*NET-SOC code the caller asked about
    soc: str                 # 6-digit SOC code of the index row
    index_title: str         # occupation title in the index
    composite: float         # 0 (protected) .. 1 (exposed)
    digital: float           # digital work intensity
    human_contact: float     # human contact intensity (protective)
    physical: float          # physical task intensity (protective)
    employment: int          # BLS OEWS national employment (0 = not reported)
    match: str               # "exact": index title == O*NET title (verified join)

    @property
    def color(self) -> str:
        return score_color(self.composite)

    @property
    def calculation(self) -> str:
        """The composite worked out from this occupation's three dimensions."""
        return (f"({self.digital:.2f} - {self.human_contact:.2f} - "
                f"{self.physical:.2f} + 2) / 3 = {self.composite:.2f}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(source=SOURCE_NAME, source_url=SOURCE_URL, color=self.color,
                 calculation=self.calculation)
        return d

    def evidence_text(self, onet_title: str) -> str:
        """One evidence line the generator can cite (unit id X-<onet_soc>)."""
        emp = (f" U.S. employment {self.employment:,} (BLS OEWS, May 2025)."
               if self.employment else "")
        return (f"AI Exposure Index ({SOURCE_NAME}, third-party, not O*NET): "
                f"{onet_title} (O*NET-SOC {self.onet_soc}) has composite "
                f"AI exposure {self.composite:.2f} on a 0-1 scale (0 = fully "
                f"protected, 1 = fully exposed). Dimensions: digital work "
                f"intensity {self.digital:.2f}, human contact intensity "
                f"{self.human_contact:.2f}, physical task intensity "
                f"{self.physical:.2f}; composite = {self.calculation} (human "
                f"contact and physical tasks protect against automation).{emp}")


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(t).lower()).strip()


class AIExposureIndex:
    def __init__(self, path: Path = INDEX_CSV):
        self.path = Path(path)
        self.rows: dict[str, dict] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    self.rows[r["soc"].strip()] = r

    @property
    def available(self) -> bool:
        return bool(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def lookup(self, onet_soc: str, onet_title: str) -> Exposure | None:
        """Index row for an O*NET occupation ("41-9041.00", "Telemarketers"),
        or None. The row must carry the same title: a 6-digit code alone is
        not enough (index row 15-2099 is Bioinformatics Technicians, O*NET
        15-2099.01, not O*NET 15-2099.00 "Mathematical Science Occupations,
        All Other")."""
        code = (onet_soc or "").strip()
        soc6 = code[:7]
        r = self.rows.get(soc6)
        if r is None:
            return None
        if _norm_title(onet_title) != _norm_title(r["title"]):
            return None
        return Exposure(
            onet_soc=code, soc=soc6, index_title=r["title"],
            composite=float(r["composite_index"]),
            digital=float(r["digital_intensity"]),
            human_contact=float(r["human_contact"]),
            physical=float(r["physical_anchor"]),
            employment=int(float(r.get("employment") or 0)),
            match="exact")

    def scores(self) -> list[float]:
        return [float(r["composite_index"]) for r in self.rows.values()]

    def score_summary(self) -> dict[str, float]:
        s = sorted(self.scores())
        if not s:
            return {}

        def q(p: float) -> float:
            return s[min(len(s) - 1, int(p * (len(s) - 1) + 0.5))]

        return {"min": s[0], "p25": q(.25), "median": q(.5), "p75": q(.75),
                "max": s[-1], "mean": round(sum(s) / len(s), 4)}


@lru_cache(maxsize=1)
def default_index() -> AIExposureIndex:
    return AIExposureIndex()


def methodology() -> dict:
    """How the scores are computed, with the references behind them."""
    return {"meaning": MEANING,
            "summary": ("Each occupation's AI exposure score comes from nine O*NET 31.0 "
                        "work-activity and work-context ratings: digital work intensity "
                        "raises exposure, human contact and physical task intensity "
                        "lower it, combined with equal weights on a 0-1 scale "
                        "(0 = fully protected, 1 = fully exposed)."),
            "formula": FORMULA, "steps": list(METHOD_STEPS),
            "variables": [{"variable": v, "onet_element": e, "scale": s,
                           "dimension": d, "direction": r}
                          for v, e, s, d, r in VARIABLES],
            "limitations": list(LIMITATIONS),
            "join": ("Joined to the chatbot's O*NET 30.3 occupations on the 6-digit "
                     "SOC code, and only when the index title equals the O*NET title, "
                     "so every score belongs to exactly the occupation it is shown for. "
                     "Occupations outside the index get no score; nothing is imputed."),
            "references": [dict(r) for r in REFERENCES]}


def about() -> dict:
    idx = default_index()
    return {"source": SOURCE_NAME, "citation": SOURCE_CITATION, "url": SOURCE_URL,
            "note": SOURCE_NOTE, "occupations": len(idx),
            "score_summary": idx.score_summary(), "methodology": methodology()}
