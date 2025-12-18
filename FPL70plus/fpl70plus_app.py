

# fpl70plus_app.py
# Streamlit app: FPL70PLUS - Smart GW Optimizer (Planning GW = Current GW + 1)
# Uses official FPL endpoints (bootstrap-static, fixtures, entry picks)
# No affiliation with Premier League/FPL.

from __future__ import annotations

import math
import time
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import numpy as np

# -----------------------------
# Config
# -----------------------------
st.set_page_config(
    page_title="FPL70PLUS — Smart GW Optimizer",
    page_icon="⚡",
    layout="wide",
)

FPL_BASE = "https://fantasy.premierleague.com/api"
UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/122.0 Safari/537.36"
}

# -----------------------------
# Helpers
# -----------------------------
def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def api_get(path: str, params: dict | None = None, timeout: int = 20) -> dict:
    url = f"{FPL_BASE}/{path.lstrip('/')}"
    r = requests.get(url, headers=UA, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

# -----------------------------
# Cached fetchers
# -----------------------------
@st.cache_data(ttl=60 * 30, show_spinner=False)  # 30 mins
def fetch_bootstrap() -> dict:
    return api_get("bootstrap-static/")

@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_fixtures() -> list:
    return api_get("fixtures/")

@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_event_live(event_id: int) -> dict:
    return api_get(f"event/{event_id}/live/")

@st.cache_data(ttl=60 * 15, show_spinner=False)
def fetch_entry(entry_id: int) -> dict:
    return api_get(f"entry/{entry_id}/")

@st.cache_data(ttl=60 * 10, show_spinner=False)
def fetch_entry_picks(entry_id: int, event_id: int) -> dict:
    # Picks for given event
    return api_get(f"entry/{entry_id}/event/{event_id}/picks/")

# -----------------------------
# Data building
# -----------------------------
def get_current_gw(bootstrap: dict) -> int:
    events = bootstrap.get("events", [])
    # "is_current" sometimes false during transitions; prefer "is_next" and "is_current"
    current = next((e for e in events if e.get("is_current")), None)
    if current:
        return int(current["id"])
    # fallback: latest started event
    started = [e for e in events if e.get("finished") or e.get("data_checked") or e.get("most_captained")]
    if started:
        return int(max(started, key=lambda x: x["id"])["id"])
    # fallback: next
    nxt = next((e for e in events if e.get("is_next")), None)
    if nxt:
        return int(nxt["id"]) - 1 if int(nxt["id"]) > 1 else 1
    return 1

def get_planning_gw(current_gw: int, bootstrap: dict) -> int:
    events = bootstrap.get("events", [])
    max_gw = int(max(e["id"] for e in events)) if events else 38
    return min(current_gw + 1, max_gw)

def build_tables(bootstrap: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    teams = pd.DataFrame(bootstrap["teams"]).copy()
    elems = pd.DataFrame(bootstrap["elements"]).copy()
    types = pd.DataFrame(bootstrap["element_types"]).copy()

    # team short name mapping
    teams["team_id"] = teams["id"].astype(int)
    teams = teams[["team_id", "name", "short_name", "strength", "strength_overall_home",
                   "strength_overall_away", "strength_attack_home", "strength_attack_away",
                   "strength_defence_home", "strength_defence_away"]]

    # position mapping
    types = types.rename(columns={"id": "element_type"})
    types["element_type"] = types["element_type"].astype(int)
    types = types[["element_type", "singular_name_short", "singular_name"]]

    elems["id"] = elems["id"].astype(int)
    elems["team"] = elems["team"].astype(int)
    elems["element_type"] = elems["element_type"].astype(int)

    elems = elems.merge(teams, left_on="team", right_on="team_id", how="left")
    elems = elems.merge(types, on="element_type", how="left")

    # basic derived fields
    elems["price"] = elems["now_cost"].astype(float) / 10.0
    elems["name"] = (elems["first_name"].fillna("") + " " + elems["second_name"].fillna("")).str.strip()

    # numeric cleaning
    for col in ["form", "points_per_game", "selected_by_percent",
                "expected_goals_per_90", "expected_assists_per_90",
                "expected_goal_involvements_per_90", "expected_goals_conceded_per_90",
                "influence", "creativity", "threat", "ict_index",
                "minutes", "total_points", "ep_next", "ep_this"]:
        if col in elems.columns:
            elems[col] = pd.to_numeric(elems[col], errors="coerce").fillna(0.0)

    # Availability fields
    # status: a=available, d=doubtful, i=injured, s=suspended, u=unavailable
    elems["status"] = elems["status"].fillna("a")
    elems["chance_next"] = pd.to_numeric(elems.get("chance_of_playing_next_round", 100), errors="coerce").fillna(100).astype(int)
    elems["chance_this"] = pd.to_numeric(elems.get("chance_of_playing_this_round", 100), errors="coerce").fillna(100).astype(int)
    elems["news"] = elems.get("news", "").fillna("")

    # Position short
    elems["pos"] = elems["singular_name_short"].fillna("UNK")

    return elems, teams, types

def build_fixtures_df(fixtures: list, teams: pd.DataFrame) -> pd.DataFrame:
    fx = pd.DataFrame(fixtures).copy()
    if fx.empty:
        return fx

    # Normalize
    fx["event"] = pd.to_numeric(fx["event"], errors="coerce").fillna(0).astype(int)
    fx["team_h"] = fx["team_h"].astype(int)
    fx["team_a"] = fx["team_a"].astype(int)

    # join team short names
    tmap = teams.set_index("team_id")["short_name"].to_dict()
    fx["home"] = fx["team_h"].map(tmap)
    fx["away"] = fx["team_a"].map(tmap)

    # results
    fx["home_goals"] = pd.to_numeric(fx.get("team_h_score"), errors="coerce")
    fx["away_goals"] = pd.to_numeric(fx.get("team_a_score"), errors="coerce")
    fx["finished"] = fx.get("finished", False).fillna(False)

    return fx

def next_fixture_label(team_id: int, planning_gw: int, fx: pd.DataFrame) -> Tuple[str, int, bool]:
    """
    Returns label like: 'BOU (H)' and FDR (1-5-ish if provided) and home? bool.
    """
    if fx.empty:
        return ("—", 3, True)
    cand = fx[(fx["event"] == planning_gw) & ((fx["team_h"] == team_id) | (fx["team_a"] == team_id))]
    if cand.empty:
        # fallback: nearest future fixture
        cand = fx[(fx["event"] > 0) & ((fx["team_h"] == team_id) | (fx["team_a"] == team_id))].sort_values("event")
        if cand.empty:
            return ("—", 3, True)
        row = cand.iloc[0]
        gw = int(row["event"])
    else:
        row = cand.iloc[0]
        gw = planning_gw

    is_home = (int(row["team_h"]) == int(team_id))
    opp = row["away"] if is_home else row["home"]
    ha = "(H)" if is_home else "(A)"
    # FDR from official "team_h_difficulty"/"team_a_difficulty" if present
    fdr = 3
    if "team_h_difficulty" in row and "team_a_difficulty" in row:
        fdr = int(row["team_h_difficulty"]) if is_home else int(row["team_a_difficulty"])
        fdr = int(_clamp(fdr, 1, 5))
    return (f"{opp} {ha}", fdr, is_home)

def compute_team_weakness(fx: pd.DataFrame, current_gw: int) -> pd.DataFrame:
    """
    Computes attack/defence weakness using actual goals from GW1..current_gw finished matches.
    Output per team_id:
      - goals_for_pg, goals_against_pg
      - att_weak (higher means weaker attack)
      - def_weak (higher means weaker defence)
    """
    if fx.empty:
        return pd.DataFrame(columns=["team_id","goals_for_pg","goals_against_pg","att_weak","def_weak"])

    hist = fx[(fx["finished"] == True) & (fx["event"] > 0) & (fx["event"] <= current_gw)].copy()
    hist = hist.dropna(subset=["home_goals","away_goals"], how="any")
    if hist.empty:
        return pd.DataFrame(columns=["team_id","goals_for_pg","goals_against_pg","att_weak","def_weak"])

    rows = []
    for _, r in hist.iterrows():
        th = int(r["team_h"]); ta = int(r["team_a"])
        hg = int(r["home_goals"]); ag = int(r["away_goals"])
        rows.append((th, hg, ag))  # home: for=hg, against=ag
        rows.append((ta, ag, hg))  # away: for=ag, against=hg

    df = pd.DataFrame(rows, columns=["team_id","gf","ga"])
    agg = df.groupby("team_id").agg(
        games=("gf","count"),
        gf=("gf","sum"),
        ga=("ga","sum"),
    ).reset_index()
    agg["goals_for_pg"] = agg["gf"] / agg["games"]
    agg["goals_against_pg"] = agg["ga"] / agg["games"]

    # Normalize weakness 0..1 using league min/max
    gf_min, gf_max = agg["goals_for_pg"].min(), agg["goals_for_pg"].max()
    ga_min, ga_max = agg["goals_against_pg"].min(), agg["goals_against_pg"].max()

    # weaker attack = lower gf_pg
    agg["att_weak"] = 1.0 - ((agg["goals_for_pg"] - gf_min) / (gf_max - gf_min + 1e-9))
    # weaker defence = higher ga_pg
    agg["def_weak"] = ((agg["goals_against_pg"] - ga_min) / (ga_max - ga_min + 1e-9))

    agg["att_weak"] = agg["att_weak"].clip(0, 1)
    agg["def_weak"] = agg["def_weak"].clip(0, 1)

    return agg[["team_id","goals_for_pg","goals_against_pg","att_weak","def_weak"]]

def risk_score_row(row: pd.Series) -> Tuple[float, str]:
    """
    Risk score 0..1 where 1 = very risky.
    Uses FPL status/chance + minutes + news signal.
    """
    status = str(row.get("status", "a"))
    chance = int(row.get("chance_next", 100))
    minutes = float(row.get("minutes", 0))
    news = str(row.get("news", "")).lower()

    # Base from chance
    base = 1.0 - (chance / 100.0)  # if chance=100 => 0 risk
    # Status penalties
    if status in ("i", "u", "s"):
        base = max(base, 0.85)
    elif status in ("d",):
        base = max(base, 0.55)
    # Very low minutes => rotation/bench risk
    # (minutes is season total; scale it)
    min_factor = 1.0 - _sigmoid((minutes - 600) / 250.0)  # <600 mins => higher
    # news keywords
    kw = 0.0
    for k in ["knock", "doubt", "ill", "injur", "suspend", "hamstring", "calf", "ankle", "out", "ruled out"]:
        if k in news:
            kw = max(kw, 0.20)
    if "expected back" in news or "should be available" in news:
        kw = min(kw, 0.10)

    score = 0.55 * base + 0.35 * min_factor + 0.10 * kw
    score = float(_clamp(score, 0.0, 1.0))

    if score >= 0.75:
        label = "Out / major doubt"
    elif score >= 0.50:
        label = "Doubt / rotation"
    elif score >= 0.30:
        label = "Some risk"
    else:
        label = "Fit"
    return score, label

def build_player_model(
    players: pd.DataFrame,
    teams_weak: pd.DataFrame,
    planning_gw: int,
    fx: pd.DataFrame,
    weights: dict,
) -> pd.DataFrame:
    """
    Builds xPts-like model for NEXT GW (planning_gw).
    Incorporates opponent weakness:
      - attackers boosted vs weak defence
      - defenders boosted vs weak attack
    """
    df = players.copy()

    # Next fixture label + opponent team_id
    opp_map = {}
    opp_id_map = {}
    fdr_map = {}
    ha_map = {}
    if not fx.empty:
        # Build helper to find opponent team id for planning gw
        for tid in df["team_id"].dropna().unique():
            tid = int(tid)
            cand = fx[(fx["event"] == planning_gw) & ((fx["team_h"] == tid) | (fx["team_a"] == tid))]
            if not cand.empty:
                r = cand.iloc[0]
                is_home = int(r["team_h"]) == tid
                opp_id = int(r["team_a"] if is_home else r["team_h"])
                opp_id_map[tid] = opp_id
                ha_map[tid] = is_home
                # label/fdr
                label, fdr, _ = next_fixture_label(tid, planning_gw, fx)
                opp_map[tid] = label
                fdr_map[tid] = fdr
            else:
                # fallback
                label, fdr, is_home = next_fixture_label(tid, planning_gw, fx)
                opp_map[tid] = label
                fdr_map[tid] = fdr
                ha_map[tid] = is_home
                # try infer opp_id from label (not perfect); set None
                opp_id_map[tid] = None

    df["next_fixture"] = df["team_id"].map(opp_map).fillna("—")
    df["next_fdr"] = df["team_id"].map(fdr_map).fillna(3).astype(int)

    # Opponent weakness join
    tw = teams_weak.set_index("team_id") if not teams_weak.empty else None
    def_weak = []
    att_weak = []
    for _, r in df.iterrows():
        tid = int(r["team_id"])
        opp_id = opp_id_map.get(tid)
        if tw is None or opp_id is None or opp_id not in tw.index:
            def_weak.append(0.5)
            att_weak.append(0.5)
        else:
            def_weak.append(float(tw.loc[opp_id, "def_weak"]))
            att_weak.append(float(tw.loc[opp_id, "att_weak"]))
    df["opp_def_weak"] = def_weak
    df["opp_att_weak"] = att_weak

    # Risk score
    rs = df.apply(risk_score_row, axis=1, result_type="expand")
    df["risk_score"] = rs[0].astype(float)
    df["risk_label"] = rs[1].astype(str)

    # Core signals (normalized-ish)
    minutes = df["minutes"].astype(float)
    form = df["form"].astype(float)
    ppg = df["points_per_game"].astype(float)
    sel = df["selected_by_percent"].astype(float)

    # attacking
    xgi90 = df.get("expected_goal_involvements_per_90", 0.0).astype(float)
    threat = df.get("threat", 0.0).astype(float)

    # fixture
    fdr = df["next_fdr"].astype(float)
    # lower fdr => easier => better
    fdr_score = 1.0 - (fdr - 1.0) / 4.0  # 1->1, 5->0

    # minutes score
    min_score = _sigmoid((minutes - 700) / 250.0)

    # Role-based opponent weakness boost
    pos = df["pos"].astype(str)
    attacker_boost = df["opp_def_weak"].astype(float)  # higher if opp defence weak
    defender_boost = df["opp_att_weak"].astype(float)  # higher if opp attack weak (clean sheet chance)

    role_boost = []
    for p, ab, db in zip(pos, attacker_boost, defender_boost):
        if p in ("FWD", "MID"):
            role_boost.append(0.6 * ab + 0.4 * 0.5)
        elif p in ("DEF", "GKP"):
            role_boost.append(0.7 * db + 0.3 * 0.5)
        else:
            role_boost.append(0.5)
    df["opp_role_boost"] = role_boost

    # Model weights (0..5 sliders)
    w_minutes = weights["minutes"]
    w_fixture = weights["fixture"]
    w_form = weights["form"]
    w_attack = weights["attack"]

    # Build xPts (simple but consistent)
    base_pts = 0.35 * ppg + 0.20 * form + 0.10 * (sel / 20.0)
    attack_pts = 0.65 * xgi90 + 0.002 * threat
    # defenders/keepers should use opp attack weakness more; already in role boost.

    raw = (
        (w_minutes * min_score)
        + (w_fixture * fdr_score)
        + (w_form * _sigmoid((form - 2.0) / 1.0))
        + (w_attack * _sigmoid((attack_pts - 0.5) / 0.35))
    )

    raw = raw / (w_minutes + w_fixture + w_form + w_attack + 1e-9)
    # scale to points range ~2..10 using base and role boost
    xpts = 2.0 + 7.5 * raw
    xpts = xpts * (0.85 + 0.30 * df["opp_role_boost"].astype(float))
    xpts = xpts + 0.35 * base_pts

    # Apply risk penalty
    xpts = xpts * (1.0 - 0.55 * df["risk_score"].astype(float))

    df["xPts"] = xpts.round(2)

    # Captain score: prefer explosive attackers with low risk and good fixture
    cap = (
        df["xPts"].astype(float)
        + 2.2 * df["opp_def_weak"].astype(float).where(df["pos"].isin(["FWD","MID"]), 0.25)
        + 1.0 * (1.0 - df["risk_score"].astype(float))
    )
    df["CaptainScore"] = cap.round(2)

    # Helpful display
    df["team_short"] = df["short_name"].fillna("—")
    return df

# -----------------------------
# Picks & squad handling
# -----------------------------
def picks_to_squad(players: pd.DataFrame, picks_json: dict) -> pd.DataFrame:
    picks = picks_json.get("picks", [])
    if not picks:
        return pd.DataFrame()

    dfp = pd.DataFrame(picks)
    dfp["element"] = dfp["element"].astype(int)
    merged = dfp.merge(players, left_on="element", right_on="id", how="left")
    # sort by position for display
    pos_order = {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}
    merged["pos_order"] = merged["pos"].map(pos_order).fillna(9).astype(int)
    merged = merged.sort_values(["multiplier", "pos_order"], ascending=[False, True]).reset_index(drop=True)
    return merged

# -----------------------------
# Transfer / Free Hit logic
# -----------------------------
def build_candidate_pool(model_df: pd.DataFrame, budget: float, allowed_positions: Optional[List[str]] = None) -> pd.DataFrame:
    cand = model_df.copy()
    # Basic filters
    cand = cand[cand["status"].isin(["a","d","i","s","u"])].copy()
    if allowed_positions:
        cand = cand[cand["pos"].isin(allowed_positions)].copy()
    # remove extreme risk
    cand = cand[cand["risk_score"] <= 0.85].copy()
    # budget screen
    cand = cand[cand["price"] <= budget + 2.0].copy()  # keep a bit wider for comparisons
    return cand.sort_values("xPts", ascending=False)

def make_transfer_suggestions(
    squad_df: pd.DataFrame,
    model_df: pd.DataFrame,
    free_transfers: int,
    bank: float,
    horizon: int = 1,
) -> pd.DataFrame:
    """
    Simple single-step suggestions repeated up to free_transfers:
    - propose OUT one weak player (low xPts, high risk) and IN best replacement same pos within budget
    """
    if squad_df.empty:
        return pd.DataFrame()

    ft = int(_clamp(free_transfers, 1, 5))
    # Start from starting XI only (multiplier>0)
    xi = squad_df[squad_df["multiplier"] > 0].copy()
    # Evaluate "badness"
    xi["bad"] = (
        (1.0 - _sigmoid((xi["xPts"].astype(float) - 4.5) / 1.5)) * 0.60
        + xi["risk_score"].astype(float) * 0.40
    )
    xi = xi.sort_values("bad", ascending=False)

    suggestions = []
    remaining_bank = float(bank)

    owned_ids = set(squad_df["id"].dropna().astype(int).tolist())

    for i in range(ft):
        if xi.empty:
            break
        out_row = xi.iloc[i % len(xi)]
        out_id = int(out_row["id"])
        out_pos = str(out_row["pos"])
        out_price = float(out_row["price"])

        max_in_price = out_price + remaining_bank
        pool = build_candidate_pool(model_df, budget=max_in_price, allowed_positions=[out_pos])
        pool = pool[~pool["id"].astype(int).isin(owned_ids)].copy()
        if pool.empty:
            continue

        best_in = pool.iloc[0]

        # compute gain
        gain = float(best_in["xPts"]) - float(out_row["xPts"])

        suggestions.append({
            "OUT": out_row["name"],
            "OUT_team": out_row["team_short"],
            "OUT_pos": out_pos,
            "OUT_price": out_price,
            "OUT_next": out_row.get("next_fixture", "—"),

            "IN": best_in["name"],
            "IN_team": best_in["team_short"],
            "IN_pos": out_pos,
            "IN_price": float(best_in["price"]),
            "IN_next": best_in.get("next_fixture", "—"),

            "NextGW_gain_xPts": round(gain, 2),
            "IN_xPts": float(best_in["xPts"]),
            "OUT_xPts": float(out_row["xPts"]),
            "IN_risk": best_in["risk_label"],
            "OUT_risk": out_row["risk_label"],
        })

        # update state (assume transfer made)
        remaining_bank = remaining_bank + out_price - float(best_in["price"])
        owned_ids.add(int(best_in["id"]))

    return pd.DataFrame(suggestions)

def make_freehit_xi(model_df: pd.DataFrame, budget_total: float = 100.0) -> pd.DataFrame:
    """
    Greedy Free Hit XI:
      1 GK, 3 DEF, 4 MID, 3 FWD (3-4-3)
    Budget approximate (does not emulate exact FPL constraints fully, but stays realistic).
    """
    df = model_df.copy()
    df = df[df["risk_score"] <= 0.70].copy()
    df = df[df["status"].isin(["a","d"])].copy()

    # position quotas
    quotas = {"GKP": 1, "DEF": 3, "MID": 4, "FWD": 3}

    picked = []
    used = set()
    spend = 0.0

    # soft price caps per slot to avoid 11 premiums
    caps = {"GKP": 6.0, "DEF": 7.0, "MID": 13.5, "FWD": 14.5}

    for pos, q in quotas.items():
        pool = df[df["pos"] == pos].sort_values("xPts", ascending=False).copy()
        for _ in range(q):
            # pick best within remaining budget and cap
            remaining = budget_total - spend
            cand = pool[(pool["price"] <= min(caps[pos], remaining)) & (~pool["id"].astype(int).isin(used))]
            if cand.empty:
                # relax cap if needed
                cand = pool[(pool["price"] <= remaining) & (~pool["id"].astype(int).isin(used))]
            if cand.empty:
                break
            row = cand.iloc[0]
            picked.append(row)
            used.add(int(row["id"]))
            spend += float(row["price"])

    if len(picked) < 11:
        return pd.DataFrame()

    out = pd.DataFrame(picked)
    out["pos_order"] = out["pos"].map({"GKP":1,"DEF":2,"MID":3,"FWD":4}).fillna(9).astype(int)
    out = out.sort_values(["pos_order","xPts"], ascending=[True, False]).reset_index(drop=True)
    out["Budget_used"] = round(spend, 2)
    out["Budget_left"] = round(budget_total - spend, 2)
    return out

# -----------------------------
# Formation renderer (centered, no HTML source)
# -----------------------------
def render_formation_grid(rows: Dict[str, List[dict]], title: str):
    """
    rows: { "GKP":[...], "DEF":[...], "MID":[...], "FWD":[...] }
    Each item dict expects: name, team_short, pos, xPts, next_fixture, risk_label, photo_url(optional)
    """
    def card_html(p: dict) -> str:
        photo = p.get("photo_url") or ""
        name = p.get("name","—")
        meta = f'{p.get("team_short","—")} · {p.get("pos","—")}'
        xpts = p.get("xPts","—")
        nxt = p.get("next_fixture","—")
        risk = p.get("risk_label","—")

        risk_class = "fit"
        if "Out" in risk:
            risk_class = "out"
        elif "Doubt" in risk:
            risk_class = "doubt"
        elif "Some" in risk:
            risk_class = "some"

        img = f'<img class="pimg" src="{photo}" />' if photo else '<div class="pimg placeholder"></div>'

        return f"""
        <div class="pcard">
          <div class="pwrap">
            {img}
            <div class="pname">{name}</div>
            <div class="pmeta">{meta}</div>
            <div class="pxpts">⭐ xPts: {xpts}</div>
            <div class="pnext">Next: {nxt}</div>
            <div class="prisk {risk_class}">{risk}</div>
          </div>
        </div>
        """

    def row_html(items: List[dict]) -> str:
        if not items:
            return '<div class="prow"></div>'
        cards = "\n".join(card_html(x) for x in items)
        return f'<div class="prow">{cards}</div>'

    html = f"""
    <div class="formation">
      <div class="ftitle">{title}</div>
      {row_html(rows.get("GKP", []))}
      {row_html(rows.get("DEF", []))}
      {row_html(rows.get("MID", []))}
      {row_html(rows.get("FWD", []))}
    </div>

    <style>
      .formation {{
        width: 100%;
        padding: 12px 10px 18px 10px;
        border-radius: 16px;
        background: rgba(10, 14, 25, 0.35);
        border: 1px solid rgba(255,255,255,0.08);
      }}
      .ftitle {{
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial;
        font-weight: 700;
        font-size: 18px;
        margin: 0 0 10px 0;
        color: rgba(255,255,255,0.92);
      }}
      .prow {{
        display: flex;
        justify-content: center;
        gap: 14px;
        margin: 12px 0;
        flex-wrap: nowrap;
      }}
      @media (max-width: 1100px) {{
        .prow {{ flex-wrap: wrap; }}
      }}
      .pcard {{
        width: 210px;
        border-radius: 16px;
        background: rgba(9, 12, 22, 0.75);
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
      }}
      .pwrap {{
        padding: 12px 10px 12px 10px;
        text-align: center;
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial;
      }}
      .pimg {{
        width: 62px; height: 62px;
        border-radius: 999px;
        object-fit: cover;
        border: 3px solid rgba(255, 85, 85, 0.9);
        background: rgba(255,255,255,0.06);
        margin: 2px auto 8px auto;
        display: block;
      }}
      .pimg.placeholder {{
        width: 62px; height: 62px;
        border-radius: 999px;
        border: 3px solid rgba(255, 85, 85, 0.6);
      }}
      .pname {{
        font-weight: 800;
        font-size: 16px;
        color: rgba(255,255,255,0.95);
        margin: 2px 0 2px 0;
      }}
      .pmeta {{
        font-size: 12px;
        color: rgba(255,255,255,0.70);
        margin-bottom: 6px;
      }}
      .pxpts {{
        font-size: 13px;
        color: rgba(255, 215, 100, 0.95);
        font-weight: 700;
        margin-bottom: 6px;
      }}
      .pnext {{
        font-size: 12px;
        color: rgba(255,255,255,0.80);
        margin-bottom: 8px;
      }}
      .prisk {{
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 800;
        border: 1px solid rgba(255,255,255,0.10);
      }}
      .prisk.fit {{ color: #9fffb0; background: rgba(40, 180, 80, 0.14); }}
      .prisk.some {{ color: #ffe59a; background: rgba(180, 140, 40, 0.16); }}
      .prisk.doubt {{ color: #ffb4b4; background: rgba(200, 70, 70, 0.18); }}
      .prisk.out {{ color: #ff8a8a; background: rgba(220, 40, 40, 0.22); }}
    </style>
    """
    components.html(html, height=640, scrolling=True)

def photo_url_from_code(code: str) -> str:
    # Official photos use `photo` like "12345.jpg"
    # common URL:
    # https://resources.premierleague.com/premierleague/photos/players/110x140/p12345.png
    # In bootstrap, `photo` is like "12345.jpg"
    try:
        base = str(code).split(".")[0]
        return f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{base}.png"
    except Exception:
        return ""

# -----------------------------
# UI
# -----------------------------
st.markdown("""
<style>
/* keep text readable even in light mode */
html, body, [class*="css"]  {
  color: rgba(255,255,255,0.92) !important;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    entry_id = st.number_input("Your FPL Entry ID", min_value=1, value=7750422, step=1)
    points_target = st.slider("🎯 Points target for next GW", 40, 95, 70, 1)
    include_bench = st.checkbox("Include bench in target (Bench Boost?)", value=False)
    free_transfers = st.slider("Free transfers this GW (1–5)", 1, 5, 1, 1)

    st.markdown("---")
    st.markdown("## 🧪 Model tuning (0–5)")
    st.caption("These sliders DO change Captain/Transfers/Free Hit.")
    w_minutes = st.slider("Minutes importance", 0, 5, 3, 1)
    w_fixture = st.slider("Fixture importance (FDR)", 0, 5, 4, 1)
    w_form = st.slider("Form importance", 0, 5, 3, 1)
    w_attack = st.slider("Attacking stats importance", 0, 5, 4, 1)

    analyse_btn = st.button("Analyse my team", type="primary")
    force_refresh = st.button("Force refresh")

if force_refresh:
    st.cache_data.clear()
    st.toast("Cache cleared. Re-fetching next run.", icon="🧹")

# -----------------------------
# Main
# -----------------------------
st.title("FPL 70+ — Smart GW Optimizer (Planning GW)")

if not analyse_btn:
    st.info("Enter your Entry ID, adjust weights if you like, then click **Analyse my team**.")
    st.stop()

t0 = time.time()
with st.spinner("Fetching FPL data..."):
    bootstrap = fetch_bootstrap()
    fixtures = fetch_fixtures()

current_gw = get_current_gw(bootstrap)
planning_gw = get_planning_gw(current_gw, bootstrap)

players_df, teams_df, _ = build_tables(bootstrap)
fx_df = build_fixtures_df(fixtures, teams_df)

# team weakness from GW1..current_gw
tw = compute_team_weakness(fx_df, current_gw)

weights = {
    "minutes": max(0.1, float(w_minutes)),
    "fixture": max(0.1, float(w_fixture)),
    "form": max(0.1, float(w_form)),
    "attack": max(0.1, float(w_attack)),
}

model_df = build_player_model(players_df, tw, planning_gw, fx_df, weights)

# Entry info + bank + picks for CURRENT GW squad, but evaluate for PLANNING GW
with st.spinner("Loading your squad..."):
    entry = fetch_entry(int(entry_id))
    picks = fetch_entry_picks(int(entry_id), int(current_gw))

bank = float(picks.get("entry_history", {}).get("bank", 0)) / 10.0

squad_df = picks_to_squad(model_df, picks)

# Summary header
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Current GW", f"{current_gw}")
c2.metric("Planning for GW", f"{planning_gw}")
c3.metric("Bank", f"£{bank:.1f}m")
c4.metric("Points target", f"{points_target} pts")
proj_xi = squad_df[squad_df["multiplier"] > 0]["xPts"].sum() if not squad_df.empty else 0.0
proj_bench = squad_df[squad_df["multiplier"] == 0]["xPts"].sum() if not squad_df.empty else 0.0
proj_total = proj_xi + (proj_bench if include_bench else 0.0)
c5.metric("Projected xPts", f"{proj_total:.2f}")

gap = points_target - proj_total
st.progress(_clamp(proj_total / max(points_target, 1), 0, 1))
st.caption(f"Planning GW projection is **{proj_total:.2f}** vs target **{points_target}**. Gap ≈ **{gap:+.2f}** points.")

with st.expander("ℹ️ What makes this version more accurate?"):
    st.write(
        "- Captain/Transfers/Free Hit are calculated for **NEXT GW (current+1)**.\n"
        "- Opponent weakness uses **real results from GW1→current GW** (goals for/against per game).\n"
        "- Risk uses FPL fields (`status`, `chance_of_playing_next_round`, `news`) + minutes trend.\n"
        "- Fixtures and team opponents come from official `/fixtures/`.\n"
        "- Model tuning sliders (0–5) directly affect xPts/Captain/Transfers/FreeHit."
    )

tabs = st.tabs([
    "🧩 Squad (Planning GW)",
    "⭐ Captaincy (Planning GW)",
    "🔁 Transfers (Planning GW)",
    "🃏 Free Hit (Planning GW)",
    "📝 Notes",
])

# -----------------------------
# Squad tab (formation)
# -----------------------------
with tabs[0]:
    if squad_df.empty:
        st.error("Could not load your squad picks. Check Entry ID.")
        st.stop()

    xi = squad_df[squad_df["multiplier"] > 0].copy()
    bench = squad_df[squad_df["multiplier"] == 0].copy()

    # Build formation rows (centered)
    rows = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
    for _, r in xi.iterrows():
        rows[str(r["pos"])].append({
            "name": r["name"],
            "team_short": r["team_short"],
            "pos": r["pos"],
            "xPts": r["xPts"],
            "next_fixture": r.get("next_fixture", "—"),
            "risk_label": r.get("risk_label", "—"),
            "photo_url": photo_url_from_code(r.get("photo", "")),
        })

    render_formation_grid(rows, title="Your XI — Planning GW (Next GW fixtures)")

    st.subheader("Bench")
    if not bench.empty:
        show = bench[["name","pos","team_short","price","xPts","next_fixture","risk_label","risk_score"]].copy()
        show = show.rename(columns={
            "team_short": "Team",
            "next_fixture": "Next fixture",
            "risk_label": "Risk",
            "risk_score": "Risk score",
        })
        st.dataframe(show, use_container_width=True, hide_index=True)
    else:
        st.caption("No bench data available.")

# -----------------------------
# Captaincy
# -----------------------------
with tabs[1]:
    # filter to realistic captain candidates: mostly attackers, low risk
    cand = squad_df[squad_df["multiplier"] > 0].copy()
    cand = cand[cand["risk_score"] <= 0.70].copy()
    cand = cand[cand["pos"].isin(["FWD","MID"])].copy()

    if cand.empty:
        st.warning("No captain candidates found after risk filtering. Lower risk strictness (sliders) or ensure your squad has fit attackers.")
    else:
        cand = cand.sort_values("CaptainScore", ascending=False)
        cap = cand.iloc[0]
        vc = cand.iloc[1] if len(cand) > 1 else cand.iloc[0]

        st.markdown("### ⭐ Captaincy — best C & VC (Planning GW)")
        st.write(
            f"**Suggested Captain (C):** `{cap['name']}` ({cap['team_short']} · {cap['pos']}) "
            f"— xPts ≈ **{cap['xPts']}**, CaptainScore ≈ **{cap['CaptainScore']}**  \n"
            f"Next: **{cap.get('next_fixture','—')}** · Risk: **{cap.get('risk_label','—')}**"
        )
        st.write(
            f"**Suggested Vice-captain (VC):** `{vc['name']}` ({vc['team_short']} · {vc['pos']}) "
            f"— xPts ≈ **{vc['xPts']}**, CaptainScore ≈ **{vc['CaptainScore']}**  \n"
            f"Next: **{vc.get('next_fixture','—')}** · Risk: **{vc.get('risk_label','—')}**"
        )

        tbl = cand[["name","pos","team_short","xPts","CaptainScore","selected_by_percent","next_fixture","risk_label","risk_score"]].copy()
        tbl = tbl.rename(columns={
            "team_short": "Team",
            "selected_by_percent": "Own%",
            "next_fixture": "Next fixture",
            "risk_label": "Risk",
            "risk_score": "Risk score",
        })
        st.dataframe(tbl.head(15), use_container_width=True, hide_index=True)

# -----------------------------
# Transfers
# -----------------------------
with tabs[2]:
    st.markdown("### 🔁 Transfer suggestions (Planning GW)")
    if squad_df.empty:
        st.warning("No squad loaded.")
    else:
        xi = squad_df[squad_df["multiplier"] > 0].copy()
        sugg = make_transfer_suggestions(
            squad_df=squad_df,
            model_df=model_df,
            free_transfers=int(free_transfers),
            bank=float(bank),
            horizon=1,
        )

        if sugg.empty:
            st.warning("No transfer suggestions found with current filters/budget.")
        else:
            # Ensure we always show exactly N rows if possible
            st.caption(f"Showing up to **{free_transfers}** suggestion(s) for next GW (GW {planning_gw}).")
            show = sugg.copy()
            show = show.rename(columns={
                "OUT_team": "OUT Team",
                "OUT_pos": "Pos",
                "OUT_price": "OUT £",
                "OUT_next": "OUT Next",
                "IN_team": "IN Team",
                "IN_price": "IN £",
                "IN_next": "IN Next",
                "NextGW_gain_xPts": "Gain xPts",
            })
            st.dataframe(show, use_container_width=True, hide_index=True)

# -----------------------------
# Free Hit
# -----------------------------
with tabs[3]:
    st.markdown("### 🃏 Free Hit draft (3–4–3) — Planning GW")
    fh = make_freehit_xi(model_df, budget_total=100.0)

    if fh.empty:
        st.warning("Could not build Free Hit XI with current risk filters. Try reducing risk strictness (or your sliders).")
    else:
        # render in formation
        rows = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
        for _, r in fh.iterrows():
            rows[str(r["pos"])].append({
                "name": r["name"],
                "team_short": r["team_short"],
                "pos": r["pos"],
                "xPts": r["xPts"],
                "next_fixture": r.get("next_fixture", "—"),
                "risk_label": r.get("risk_label", "—"),
                "photo_url": photo_url_from_code(r.get("photo", "")),
            })
        render_formation_grid(rows, title=f"Free Hit XI (3–4–3) — GW {planning_gw}")

        st.caption(f"Budget used: £{fh['Budget_used'].iloc[0]:.2f}m · Left: £{fh['Budget_left'].iloc[0]:.2f}m")
        st.dataframe(
            fh[["name","pos","team_short","price","xPts","next_fixture","risk_label","risk_score"]]
              .rename(columns={"team_short":"Team","next_fixture":"Next fixture","risk_label":"Risk","risk_score":"Risk score"}),
            use_container_width=True,
            hide_index=True
        )

# -----------------------------
# Notes
# -----------------------------
with tabs[4]:
    st.markdown("### 📝 Notes / Debug")
    st.write(f"- Data fetch time: **{(time.time()-t0):.2f}s** (cache helps a lot).")
    st.write(f"- Planning GW: **{planning_gw}** (Current GW {current_gw} + 1).")
    st.write("- If you see a wrong fixture: it usually means the fixture list changed; click **Force refresh**.")
    st.write("- If risk is too strict and blocks captain/free hit: reduce sliders or set minutes importance higher.")
