from __future__ import annotations

import html
import json
import re
import sqlite3
from datetime import UTC, date, datetime

import pandas as pd
import streamlit as st

from return_risk.config import MODELS_DIR
from return_risk.dashboard import (
    batch_distribution_items,
    demo_batch_frame,
    merchant_risk_report_html,
    prepare_batch_review_frame,
)
from return_risk.monitoring import (
    OutcomeAlreadyRecordedError,
    PredictionNotFoundError,
    ShadowMonitoringStore,
)
from return_risk.policy import (
    CostAssumptions,
    evaluate_policy_frontier,
    load_policy_frontier,
)
from return_risk.schemas import OrderRequest
from return_risk.scoring import FrozenReturnRiskScorer

st.set_page_config(
    page_title="Return Risk Manager",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
    :root {--ink:#10213e; --muted:#64748b; --line:#d8e2ef; --blue:#2f67ef;
        --navy:#0b1d42; --mint:#65d9c8; --amber:#e8a324; --surface:#f4f7fb;}
    html, body, [class*="css"] {font-family:Inter,ui-sans-serif,system-ui,-apple-system,
        BlinkMacSystemFont,"Segoe UI",sans-serif;}
    .stApp {background:radial-gradient(circle at 4% 1%,rgba(47,103,239,.24),transparent 27%),
        radial-gradient(circle at 96% 7%,rgba(40,196,186,.16),transparent 24%),
        linear-gradient(180deg,#071226 0,#0b1830 46%,#0d1d37 100%);}
    .block-container {max-width:1220px; padding-top:1rem; padding-bottom:3rem;}
    [data-testid="stToolbar"], .stAppDeployButton, #MainMenu, footer {display:none!important;}
    [data-testid="stHeader"] {background:transparent;}
    [data-testid="stMetric"] {position:relative; overflow:hidden;
        background:linear-gradient(145deg,#f7faff,#e8f0ff); border:1px solid #c9d9f1;
        border-top:3px solid #7da5f8; border-radius:12px; padding:11px 13px;
        box-shadow:0 7px 18px rgba(28,57,109,.065);}
    [data-testid="stHorizontalBlock"]>[data-testid="stColumn"]:nth-child(2)
        [data-testid="stMetric"] {background:linear-gradient(145deg,#f3fcfa,#dff5ef);
        border-color:#bfe2d9; border-top-color:#36b9a6;}
    [data-testid="stHorizontalBlock"]>[data-testid="stColumn"]:nth-child(3)
        [data-testid="stMetric"] {background:linear-gradient(145deg,#fffaf0,#ffedcd);
        border-color:#ebd2a8; border-top-color:#e8a324;}
    [data-testid="stHorizontalBlock"]>[data-testid="stColumn"]:nth-child(4)
        [data-testid="stMetric"] {background:linear-gradient(145deg,#faf8ff,#ebe5ff);
        border-color:#d5caf3; border-top-color:#8a6fd1;}
    [data-testid="stMetricLabel"] {color:var(--muted); font-size:.84rem;}
    [data-testid="stMetricValue"] {color:var(--ink); font-weight:600;}
    [data-testid="stWidgetLabel"] p {font-size:.88rem;}
    [data-testid="stCaptionContainer"] p {font-size:.8rem; line-height:1.5;}
    [data-testid="stNumberInputContainer"] {overflow:hidden; background:#fff!important;
        border:1px solid #9fb5d3!important; border-radius:10px!important;
        box-shadow:0 3px 10px rgba(29,63,119,.10)!important;}
    [data-testid="stNumberInputContainer"]:focus-within {border-color:#2f67ef!important;
        box-shadow:0 0 0 3px rgba(47,103,239,.14)!important;}
    [data-testid="stNumberInputField"] {background:#fff!important; color:#10213e!important;
        font-weight:550;}
    [data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] {
        color:#214575!important; background:#eaf1fb!important;}
    [data-testid="stNumberInputStepDown"]:hover,
    [data-testid="stNumberInputStepUp"]:hover {background:#dce8fa!important;}
    [data-testid="stFileUploaderDropzone"] {background:rgba(255,255,255,.82)!important;
        border:1px dashed #9fb5d3!important; border-radius:11px!important;}
    [data-testid="stVerticalBlockBorderWrapper"] {border-color:var(--line)!important;
        border-radius:17px!important; background:linear-gradient(145deg,#f9fcff,#ebf3ff);
        box-shadow:0 10px 30px rgba(23,42,80,.08);}
    [data-testid="stForm"] {border:0; padding:0;}
    [data-testid="stFormSubmitButton"] button {width:100%; min-height:2.85rem;
        border-radius:10px; background:linear-gradient(135deg,#316bff,#1f55de);
        box-shadow:0 10px 22px rgba(49,107,255,.20);}
    [data-testid="stFormSubmitButton"] button:hover {border-color:#1748d3;}
    .rr-topbar {display:flex; align-items:center; justify-content:space-between; gap:18px;
        padding:10px 3px 13px;}
    .rr-brand {display:flex; align-items:center; gap:11px; color:#f8fbff;}
    .rr-brand-mark {display:grid; place-items:center; width:36px; height:36px;
        border-radius:11px; color:white; background:linear-gradient(145deg,#316bff,#1748d3);
        box-shadow:0 8px 18px rgba(49,107,255,.24); font-size:1.05rem;}
    .rr-brand strong,.rr-brand span {display:block;}
    .rr-brand strong {font-size:1rem; letter-spacing:.035em;}
    .rr-brand span {margin-top:1px; color:#b4c2d6; font-size:.78rem;}
    .rr-status {display:flex; align-items:center; gap:8px; padding:8px 11px;
        border:1px solid rgba(232,163,36,.35); border-radius:999px; color:#ffd47c;
        background:rgba(232,163,36,.10); font-size:.8rem; font-weight:700;}
    .rr-status-dot {width:8px; height:8px; border-radius:50%; background:#e8a324;
        box-shadow:0 0 0 4px rgba(232,163,36,.14);}
    .hero {display:grid; grid-template-columns:minmax(0,1.35fr) minmax(280px,.65fr);
        gap:26px; align-items:center; overflow:hidden; position:relative;
        background:radial-gradient(circle at 88% 15%,rgba(101,217,200,.17),transparent 27%),
        linear-gradient(125deg,#0b1d42,#173b82 62%,#2056bd); color:white;
        padding:34px 38px; border-radius:20px; margin-bottom:15px;
        box-shadow:0 18px 44px rgba(12,35,79,.16);}
    .hero .eyebrow {font-size:.76rem; letter-spacing:.13em; color:#85efe0;
        font-weight:700; text-transform:uppercase;}
    .hero h1 {max-width:680px; font-size:clamp(2rem,4vw,3.15rem); letter-spacing:-.045em;
        line-height:1.04; margin:9px 0 12px; color:white;}
    .hero p {max-width:650px; color:rgba(238,246,255,.82); margin:0;
        font-size:1.02rem; line-height:1.6;}
    .rr-flow {display:grid; grid-template-columns:repeat(3,1fr); gap:7px; margin-top:22px;}
    .rr-step {padding:10px 11px; border:1px solid rgba(255,255,255,.14);
        border-radius:10px; background:rgba(255,255,255,.07); font-size:.76rem;}
    .rr-step b {display:block; margin-bottom:3px; color:#85efe0; font-size:.72rem;}
    .rr-proof {padding:18px; border:1px solid rgba(255,255,255,.16); border-radius:15px;
        background:rgba(5,17,43,.34); backdrop-filter:blur(10px);}
    .rr-proof-label {color:rgba(255,255,255,.62); font-size:.7rem;
        letter-spacing:.09em; text-transform:uppercase;}
    .rr-proof-grid {display:grid; grid-template-columns:repeat(3,1fr); gap:10px;
        margin-top:13px;}
    .rr-proof-grid strong,.rr-proof-grid span {display:block;}
    .rr-proof-grid strong {color:white; font-size:1.08rem;}
    .rr-proof-grid span {margin-top:2px; color:rgba(255,255,255,.68); font-size:.72rem;}
    .rr-shadow-explainer {display:flex; align-items:center; justify-content:space-between;
        gap:18px; margin:0 0 15px; padding:14px 17px; border:1px solid #365279;
        border-left:4px solid #e8a324; border-radius:13px; color:#c9d7eb;
        background:linear-gradient(110deg,rgba(18,39,73,.96),rgba(12,31,59,.96));
        box-shadow:0 10px 25px rgba(0,0,0,.16);}
    .rr-shadow-explainer b {display:block; margin-bottom:3px; color:#fff; font-size:.9rem;}
    .rr-shadow-explainer span {font-size:.8rem; line-height:1.5;}
    .rr-shadow-pill {flex:0 0 auto; padding:7px 10px; border-radius:999px;
        color:#ffd47c; background:rgba(232,163,36,.12); font-size:.72rem!important;
        font-weight:750; white-space:nowrap;}
    .rr-section-head {margin:4px 0 17px;}
    .rr-section-head h2 {margin:0; color:var(--ink); font-size:1.45rem;
        letter-spacing:-.025em;}
    .rr-section-head p {margin:5px 0 0; color:var(--muted); font-size:.9rem;}
    .rr-panel-head {display:flex; align-items:center; gap:10px; margin-bottom:8px;}
    .rr-panel-icon {display:grid; place-items:center; width:30px; height:30px;
        border-radius:9px; background:#edf3ff; color:#245de1; font-size:.9rem;}
    .rr-panel-head b {font-size:1rem; color:var(--ink);}
    .rr-panel-caption {color:var(--muted); font-size:.82rem; margin:-2px 0 16px;}
    .rr-empty {display:flex; min-height:420px; flex-direction:column; align-items:center;
        justify-content:center; padding:25px; text-align:center;
        background:radial-gradient(circle at 50% 38%,rgba(47,103,239,.08),transparent 31%);}
    .rr-empty-orb {display:grid; place-items:center; width:72px; height:72px;
        margin-bottom:16px; border-radius:50%; color:#2f67ef; background:#edf3ff;
        border:8px solid #f6f9ff; font-size:1.45rem;}
    .rr-empty h3 {margin:0; color:var(--ink); font-size:1.05rem;}
    .rr-empty p {max-width:310px; margin:7px 0 17px; color:var(--muted);
        font-size:.85rem; line-height:1.55;}
    .rr-mini-flow {display:flex; flex-wrap:wrap; justify-content:center; gap:7px;}
    .rr-mini-flow span {padding:6px 9px; border:1px solid var(--line); border-radius:999px;
        color:#52627a; background:#fff; font-size:.75rem;}
    .rr-batch-guide {display:grid; grid-template-columns:repeat(3,minmax(0,1fr));
        gap:9px; margin:0 0 17px;}
    .rr-batch-step {padding:11px 13px; border:1px solid #cbd9ec; border-radius:11px;
        color:#52627a; background:rgba(255,255,255,.58); font-size:.76rem; line-height:1.45;}
    .rr-batch-step b {display:block; margin-bottom:3px; color:#245de1; font-size:.7rem;
        letter-spacing:.04em;}
    .rr-exposure-formula {margin:11px 0 14px; padding:10px 13px;
        border:1px solid #bcded8; border-radius:10px; color:#315d57;
        background:#effaf7; font-size:.79rem; line-height:1.5;}
    .rr-exposure-formula b {color:#135f53;}
    .rr-result {padding:3px 1px 1px;}
    .rr-score-zone {display:grid; grid-template-columns:145px minmax(0,1fr); gap:17px;
        align-items:center; margin:15px 0 18px;}
    .rr-gauge {position:relative; display:grid; place-items:center; width:145px; height:145px;
        border-radius:50%;}
    .rr-gauge:before {content:""; position:absolute; inset:13px; border-radius:50%;
        background:white;}
    .rr-score {position:relative; z-index:1; text-align:center;}
    .rr-score strong,.rr-score span {display:block;}
    .rr-score strong {color:var(--ink); font-size:1.85rem; letter-spacing:-.04em;}
    .rr-score span {margin-top:2px; color:var(--muted); font-size:.72rem;
        text-transform:uppercase; letter-spacing:.08em;}
    .rr-verdict {display:inline-block; padding:6px 8px; border-radius:8px;
        font-size:.72rem; font-weight:700;}
    .rr-verdict-high {color:#8b5b00; background:#fff7e6;}
    .rr-verdict-low {color:#08775e; background:#ecfdf5;}
    .rr-result-copy h3 {margin:9px 0 5px; color:var(--ink); font-size:1.05rem;}
    .rr-result-copy p {margin:0; color:var(--muted); font-size:.8rem; line-height:1.5;}
    .rr-fact-list {display:grid; gap:8px; padding:15px 0; margin:4px 0 14px;
        border-top:1px solid var(--line); border-bottom:1px solid var(--line);}
    .rr-fact {display:flex; justify-content:space-between; gap:12px; color:var(--muted);
        font-size:.79rem;}
    .rr-fact b {color:var(--ink);}
    .decision {display:flex; gap:10px; background:#effaf7; border:1px solid #b9e3da;
        border-radius:12px; padding:12px 14px; color:#245e55; margin:10px 0; font-size:.82rem;}
    .rr-action-grid {display:grid; grid-template-columns:repeat(2,minmax(0,1fr));
        gap:10px; margin:13px 0 5px;}
    .rr-action-card {min-width:0; padding:13px 14px; border:1px solid #c8d8ed;
        border-top:3px solid #6f99ed; border-radius:12px;
        background:linear-gradient(145deg,#f8fbff,#e9f1ff);}
    .rr-action-card span,.rr-action-card strong,.rr-action-card small {display:block;}
    .rr-action-card span {color:var(--muted); font-size:.75rem; font-weight:650;}
    .rr-action-card strong {margin:5px 0 4px; color:var(--ink); font-size:1rem;
        line-height:1.25; overflow-wrap:anywhere;}
    .rr-action-card small {color:#66758b; font-size:.75rem; line-height:1.45;}
    .rr-action-routine {border-top-color:#36b9a6;
        background:linear-gradient(145deg,#f5fffc,#e3f7f2);}
    .rr-action-review {border-top-color:#e8a324;
        background:linear-gradient(145deg,#fffaf0,#ffedcd);}
    .rr-action-warning {border-top-color:#d9783d;
        background:linear-gradient(145deg,#fff8f3,#fde9dd);}
    .rr-action-operational {border-top-color:#36b9a6;
        background:linear-gradient(145deg,#f3fcfa,#dff5ef);}
    .rr-reason-list {display:grid; gap:9px; margin:9px 0;}
    .rr-score-explain-summary {margin:8px 0 13px; padding:12px 14px;
        border:1px solid #c7d9f5; border-left:4px solid #2f67ef; border-radius:10px;
        color:#30445f; background:#f2f7ff; font-size:.82rem; line-height:1.5;}
    .rr-score-explain-summary b {color:var(--ink);}
    .rr-reason-card {padding:12px 13px; border:1px solid #cfdaea; border-radius:11px;
        background:rgba(255,255,255,.72);}
    .rr-reason-head {display:flex; justify-content:space-between; gap:10px;
        align-items:flex-start; margin-bottom:6px;}
    .rr-reason-head strong {color:var(--ink); font-size:.86rem;}
    .rr-reason-head span {flex:0 0 auto; padding:3px 6px; border-radius:999px;
        font-size:.7rem; font-weight:700;}
    .rr-effect-raised {color:#8b5b00; background:#fff1cf;}
    .rr-effect-lowered {color:#08775e; background:#ddf7f0;}
    .rr-reason-value {margin-bottom:5px; color:#40516a; font-size:.77rem;}
    .rr-reason-card p {margin:0; color:#617087; font-size:.76rem; line-height:1.55;}
    .evidence-note {border-left:4px solid var(--blue); background:#f8fafc;
        padding:12px 15px; border-radius:0 9px 9px 0; color:var(--ink);}
    .allow {background:#ecfdf5; color:#047857; border:1px solid #6ee7b7;
        border-radius:10px; padding:12px 14px;}
    .block {background:#fff1f2; color:#9f1239; border:1px solid #fda4af;
        border-radius:10px; padding:12px 14px;}
    div[data-testid="stTabs"] [role="tab"] {min-height:42px; margin-bottom:3px;
        padding:.6rem .8rem;
        border:1px solid transparent; border-radius:10px; color:#c8d4e5!important;
        font-size:.9rem; font-weight:650; transition:background .18s ease,
        border-color .18s ease,box-shadow .18s ease;}
    div[data-testid="stTabs"] [role="tab"] p {color:inherit!important;
        font-size:inherit!important;}
    div[data-testid="stTabs"] [role="tab"]:hover {color:#fff!important;
        border-color:rgba(142,172,231,.32); background:rgba(255,255,255,.07);}
    div[data-testid="stTabs"] [role="tab"][aria-selected="true"] {color:#fff!important;
        border-color:#6f99ed; background:linear-gradient(135deg,#2857c8,#173b82);
        box-shadow:0 8px 20px rgba(24,67,163,.34),inset 0 1px 0 rgba(255,255,255,.12);}
    div[data-testid="stTabs"] .react-aria-SelectionIndicator {display:none!important;}
    div[data-testid="stTabs"] [role="tablist"] {gap:.35rem; padding-bottom:6px;
        border-bottom:1px solid rgba(128,153,194,.42);}
    div[data-testid="stTabs"] [data-baseweb="tab-border"] {background-color:var(--line);}
    div[data-testid="stTabs"] [role="tabpanel"] {margin-top:12px; padding:22px 24px 28px;
        border:1px solid #bfd0e7; border-top:4px solid #6e95ed; border-radius:18px;
        background:linear-gradient(145deg,#e6eefb 0%,#eaf6f4 58%,#f0ebfb 100%);
        box-shadow:0 18px 44px rgba(31,56,98,.10);}
    .rr-trust-strip {display:grid; grid-template-columns:repeat(4,1fr); gap:1px;
        overflow:hidden; margin-top:17px; border:1px solid var(--line); border-radius:13px;
        background:var(--line);}
    .rr-trust-item {padding:12px 14px; background:#fbfcfe;}
    .rr-trust-item b,.rr-trust-item span {display:block;}
    .rr-trust-item b {color:var(--ink); font-size:.76rem;}
    .rr-trust-item span {margin-top:2px; color:var(--muted); font-size:.72rem;}
    .rr-chart-shell {padding:16px 17px 10px; border:1px solid var(--line);
        border-radius:14px; background:#fff; box-shadow:0 5px 18px rgba(15,23,42,.035);}
    .rr-chart-shell svg {display:block; width:100%; height:auto; overflow:visible;}
    .rr-chart-axis {fill:#718096; font-size:11px;}
    .rr-chart-grid {stroke:#dfe6f0; stroke-width:1; stroke-dasharray:4 5;}
    .rr-chart-zero {stroke:#e29b20; stroke-width:1.2; stroke-dasharray:5 4;}
    .rr-chart-line {fill:none; stroke:#2f67ef; stroke-width:3;
        stroke-linecap:round; stroke-linejoin:round;}
    .rr-viz-grid {display:grid; grid-template-columns:repeat(2,minmax(0,1fr));
        gap:14px; margin:14px 0;}
    .rr-mini-chart {padding:15px 16px; border:1px solid var(--line); border-radius:13px;
        background:#fff;}
    .rr-mini-chart h4 {margin:0 0 13px; color:var(--ink); font-size:.84rem;}
    .rr-bar-row {display:grid; grid-template-columns:95px minmax(0,1fr) 58px;
        gap:9px; align-items:center; margin:8px 0;}
    .rr-bar-label {overflow:hidden; color:#5e6c82; font-size:.72rem;
        text-overflow:ellipsis; white-space:nowrap;}
    .rr-bar-track {height:8px; overflow:hidden; border-radius:999px; background:#edf2f8;}
    .rr-bar-fill {height:100%; border-radius:999px;
        background:linear-gradient(90deg,#316bff,#65a4ff);}
    .rr-bar-value {color:#43516a; font-size:.72rem; text-align:right;}
    .rr-judge-grid {display:grid; grid-template-columns:repeat(3,minmax(0,1fr));
        gap:13px; margin:16px 0;}
    .rr-judge-card {padding:17px 18px; border:1px solid var(--line); border-radius:14px;
        background:#fff; box-shadow:0 6px 20px rgba(15,23,42,.04);}
    .rr-judge-card span {display:block; color:#2f67ef; font-size:.72rem; font-weight:750;
        letter-spacing:.08em; text-transform:uppercase;}
    .rr-judge-card b {display:block; margin:7px 0 5px; color:var(--ink); font-size:.92rem;}
    .rr-judge-card p {margin:0; color:var(--muted); font-size:.8rem; line-height:1.55;}
    .rr-judge-card:nth-child(1) {background:linear-gradient(145deg,#fff,#eef4ff);
        border-color:#cdddfb;}
    .rr-judge-card:nth-child(2) {background:linear-gradient(145deg,#fff,#ecfaf7);
        border-color:#c9ebe4;}
    .rr-judge-card:nth-child(3) {background:linear-gradient(145deg,#fff,#fff6e7);
        border-color:#f1dbb4;}
    .rr-architecture {display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:8px;
        margin:13px 0 7px;}
    .rr-architecture div {position:relative; min-height:72px; padding:12px 10px;
        border:1px solid #c6d7f0; border-radius:11px;
        background:linear-gradient(145deg,#f5f9ff,#e2edff);
        color:#44546c; font-size:.76rem; line-height:1.4;}
    .rr-architecture b {display:block; margin-bottom:5px; color:#255bd7; font-size:.7rem;}
    .rr-preset-row {margin:0 0 13px; padding:13px 14px; border:1px solid var(--line);
        border-radius:13px; background:linear-gradient(120deg,#eef4ff,#f4fbfa);}
    .rr-feature-hint {display:flex; gap:12px; align-items:flex-start; margin:0 0 17px;
        padding:13px 15px; border:1px solid; border-radius:13px;}
    .rr-feature-hint .icon {display:grid; place-items:center; flex:0 0 30px; width:30px;
        height:30px; border-radius:9px; font-weight:750;}
    .rr-feature-hint b,.rr-feature-hint span {display:block;}
    .rr-feature-hint b {font-size:.86rem; color:var(--ink);}
    .rr-feature-hint span {margin-top:2px; color:#5f6f86; font-size:.78rem; line-height:1.5;}
    .rr-hint-blue {background:#eef4ff; border-color:#cbdcfb;}
    .rr-hint-blue .icon {background:#dce8ff; color:#245de1;}
    .rr-hint-mint {background:#edf9f7; border-color:#c7e9e2;}
    .rr-hint-mint .icon {background:#d9f3ee; color:#08775e;}
    .rr-hint-amber {background:#fff7e9; border-color:#efd8af;}
    .rr-hint-amber .icon {background:#ffebc5; color:#915a00;}
    .rr-hint-slate {background:#f2f5f9; border-color:#d8e1ec;}
    .rr-hint-slate .icon {background:#e4eaf2; color:#40516a;}
    [data-testid="stDownloadButton"] button {border-color:#bddfd9;
        background:linear-gradient(135deg,#f0fcf9,#e3f7f3); color:#126653;}
    @media (max-width:900px) {.hero {grid-template-columns:1fr; padding:28px;}
        .rr-score-zone {grid-template-columns:1fr; justify-items:center; text-align:center;}
        .rr-architecture {grid-template-columns:repeat(3,1fr);}}
    @media (max-width:700px) {.block-container {padding-left:.8rem; padding-right:.8rem;}
        .rr-topbar {align-items:stretch; flex-direction:column;}
        .rr-status {align-self:flex-start; font-size:.7rem; padding:7px 9px;}
        .hero {padding:24px 19px;} .hero h1 {font-size:2rem;}
        .rr-shadow-explainer {align-items:flex-start; flex-direction:column;}
        .rr-flow,.rr-trust-strip,.rr-viz-grid,.rr-judge-grid,.rr-batch-guide,
        .rr-action-grid,
        .rr-architecture {grid-template-columns:1fr;}
        .rr-proof-grid {grid-template-columns:repeat(3,minmax(0,1fr));}}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_scorer() -> FrozenReturnRiskScorer:
    return FrozenReturnRiskScorer.from_project()


@st.cache_data
def load_runtime_evidence() -> tuple[dict, dict]:
    frontier = load_policy_frontier(MODELS_DIR / "policy_frontier.json")
    selection = json.loads(
        (MODELS_DIR / "model_selection_summary.json").read_text(encoding="utf-8")
    )
    if selection.get("test_set_accessed_for_selection") is not False:
        raise RuntimeError("Model-selection summary is not validation-only evidence.")
    return frontier, selection


def readable_feature(name: str) -> str:
    labels = {
        "Product_Category": "Product category",
        "Discount_Applied": "Discount",
        "order_year": "Order date pattern",
    }
    return labels.get(name, name.replace("_", " ").capitalize())


def plain_reason(technical_reason: str, direction: str) -> str:
    """Translate auditable model evidence into judge-friendly language."""
    comparison = re.search(
        r"returned at (?P<rate>[\d.]+%) versus (?P<overall>[\d.]+%) overall",
        technical_reason,
    )
    effect = "raised" if direction == "raised" else "lowered"
    if comparison:
        rate = comparison.group("rate")
        overall = comparison.group("overall")
        frequency = "more" if float(rate.rstrip("%")) > float(overall.rstrip("%")) else "less"
        return (
            f"Similar orders returned {frequency} often than average in past data—"
            f"{rate} compared with {overall}. Therefore, this factor {effect} the risk score."
        )
    if technical_reason.startswith("No ") and "used in development" in technical_reason:
        return (
            "This date is outside the period used to develop the model, so its effect is "
            "less certain. Treat it as a time-pattern warning, not a cause of returns."
        )
    if technical_reason.startswith("No development orders fell"):
        return (
            "There were too few similar past orders to explain this factor confidently. "
            f"The model still {effect} the score, but this signal is less reliable."
        )
    return (
        f"This factor {effect} the score based on patterns learned from past development "
        "data. It does not mean the factor causes a return."
    )


def policy_frontier_chart(frame: pd.DataFrame) -> str:
    x_values = frame["Review capacity used (%)"].to_numpy(dtype=float)
    y_values = frame["Savings per 1,000 (₹)"].to_numpy(dtype=float)
    left, right, top, bottom = 54.0, 706.0, 18.0, 184.0
    x_min, x_max = float(x_values.min()), float(x_values.max())
    y_min, y_max = float(y_values.min()), float(y_values.max())
    if x_max == x_min:
        x_max = x_min + 1
    if y_max == y_min:
        y_max = y_min + 1

    def x_position(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (right - left)

    def y_position(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    points = " ".join(
        f"{x_position(x_value):.1f},{y_position(y_value):.1f}"
        for x_value, y_value in zip(x_values, y_values, strict=True)
    )
    zero_line = ""
    if y_min <= 0 <= y_max:
        zero_y = y_position(0)
        zero_line = (
            f'<line class="rr-chart-zero" x1="{left}" y1="{zero_y:.1f}" '
            f'x2="{right}" y2="{zero_y:.1f}" />'
        )
    return f"""
    <div class="rr-chart-shell">
      <svg viewBox="0 0 760 220" role="img"
      aria-label="Validation savings by review capacity used">
        <line class="rr-chart-grid" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" />
        <line class="rr-chart-grid" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" />
        {zero_line}
        <polyline class="rr-chart-line" points="{points}" />
        <text class="rr-chart-axis" x="{left}" y="208">{x_min:.0f}% review</text>
        <text class="rr-chart-axis" x="{right}" y="208" text-anchor="end">{x_max:.0f}% review</text>
        <text class="rr-chart-axis" x="{left - 7}" y="{top + 4}" text-anchor="end">
        ₹{y_max:,.0f}</text>
        <text class="rr-chart-axis" x="{left - 7}" y="{bottom + 4}" text-anchor="end">
        ₹{y_min:,.0f}</text>
      </svg>
    </div>
    """


def horizontal_bar_chart(title: str, items: list[tuple[str, int, str]]) -> str:
    maximum = max((value for _, value, _ in items), default=1)
    rows = []
    for label, value, detail in items:
        width = value / max(1, maximum) * 100
        rows.append(
            '<div class="rr-bar-row">'
            f'<span class="rr-bar-label" title="{html.escape(label)}">'
            f"{html.escape(label)}</span>"
            '<span class="rr-bar-track">'
            f'<span class="rr-bar-fill" style="display:block;width:{width:.1f}%"></span>'
            "</span>"
            f'<span class="rr-bar-value">{html.escape(detail)}</span>'
            "</div>"
        )
    return (
        '<section class="rr-mini-chart">'
        f"<h4>{html.escape(title)}</h4>{''.join(rows)}</section>"
    )


def feature_hint(tone: str, icon: str, title: str, description: str) -> None:
    st.markdown(
        f'<div class="rr-feature-hint rr-hint-{html.escape(tone)}">'
        f'<span class="icon">{html.escape(icon)}</span><div>'
        f'<b>{html.escape(title)}</b><span>{html.escape(description)}</span>'
        "</div></div>",
        unsafe_allow_html=True,
    )


def record_score(store: ShadowMonitoringStore, result, source: str) -> str:
    return store.record_prediction(
        release_id=result.release_id,
        source=source,
        risk_score=result.risk_score,
        decision_threshold=result.decision_threshold,
        would_flag=result.would_flag_under_frozen_policy,
        computed_order_value=result.computed_order_value,
    )


DEMO_PRESETS = {
    "routine": {
        "order_product_category": "Books",
        "order_product_price": 300.0,
        "order_quantity": 1,
        "order_discount": 0.0,
        "order_shipping": "Express",
        "order_payment": "Credit Card",
        "order_date": date(2024, 12, 1),
        "demo_preset_notice": "Routine example loaded — expected to remain below the review line.",
    },
    "elevated": {
        "order_product_category": "Clothing",
        "order_product_price": 1200.0,
        "order_quantity": 1,
        "order_discount": 50.0,
        "order_shipping": "Standard",
        "order_payment": "COD",
        "order_date": date(2024, 12, 1),
        "demo_preset_notice": "Elevated example loaded — expected to cross the frozen review line.",
    },
    "warning": {
        "order_product_category": "Electronics",
        "order_product_price": 5000.0,
        "order_quantity": 2,
        "order_discount": 45.0,
        "order_shipping": "Next-Day",
        "order_payment": "Wallet",
        "order_date": date(2026, 8, 30),
        "demo_preset_notice": (
            "Out-of-range example loaded — scoring should display reliability warnings."
        ),
    },
}

DEFAULT_ORDER_INPUTS = {
    "order_product_category": "Clothing",
    "order_product_price": 1200.0,
    "order_quantity": 1,
    "order_discount": 25.0,
    "order_shipping": "Standard",
    "order_payment": "Credit Card",
    "order_date": date(2024, 12, 1),
}


def apply_demo_preset(name: str) -> None:
    for key, value in DEMO_PRESETS[name].items():
        st.session_state[key] = value


scorer = load_scorer()
monitoring_store = ShadowMonitoringStore.from_environment()
card = scorer.model_card()
metrics = card["held_out_metrics"]
policy_frontier, model_selection = load_runtime_evidence()

st.markdown(
    f"""
    <div class="rr-topbar">
      <div class="rr-brand">
        <div class="rr-brand-mark">◆</div>
        <div><strong>Return Risk Manager</strong>
        <span>Razorpay Buildathon · AI Risk Manager</span></div>
      </div>
      <div class="rr-status"><span class="rr-status-dot"></span>
      <span>SHADOW MODE · SCORES ONLY</span></div>
    </div>
    <div class="hero">
      <div>
        <div class="eyebrow">Pre-shipment risk intelligence</div>
        <h1>Know which orders may come back—before they ship.</h1>
        <p>Score return probability using checkout-time information, explain the result,
        and learn from completed outcomes without adding unsafe customer friction.</p>
        <div class="rr-flow">
          <div class="rr-step"><b>01 · INPUT</b>Checkout details</div>
          <div class="rr-step"><b>02 · DETECT</b>Return-risk score</div>
          <div class="rr-step"><b>03 · RESPOND</b>Safe merchant action</div>
        </div>
      </div>
      <div class="rr-proof">
        <div class="rr-proof-label">Held-out test evidence</div>
        <div class="rr-proof-grid">
          <div><strong>{metrics['orders']:,}</strong><span>test orders</span></div>
          <div><strong>{metrics['precision']:.1%}</strong><span>precision</span></div>
          <div><strong>{metrics['recall']:.1%}</strong><span>recall</span></div>
        </div>
      </div>
    </div>
    <div class="rr-shadow-explainer">
      <div><b>What shadow mode means</b>
      <span>The model scores and explains orders, then records outcomes for evaluation—but it
      cannot block or delay an order, add customer verification, or change return rights. This
      safety gate is active because held-out intervention economics were negative.</span></div>
      <span class="rr-shadow-pill">NO CUSTOMER ACTION</span>
    </div>
    """,
    unsafe_allow_html=True,
)

overview_tab, detector_tab, batch_tab, monitoring_tab, policy_tab, evidence_tab = st.tabs(
    [
        "Judge overview",
        "Try the detector",
        "Batch risk review",
        "Monitor outcomes",
        "Policy Lab",
        "Model evidence",
    ]
)

with overview_tab:
    st.markdown(
        '<div class="rr-section-head"><h2>The complete case in 60 seconds</h2>'
        "<p>A working, defense-only return-risk detector with frozen held-out evidence "
        "and an economic safety gate.</p></div>",
        unsafe_allow_html=True,
    )
    feature_hint(
        "blue",
        "01",
        "Purpose of this page",
        "Understand the detector, its held-out evidence, and its safety decision in 60 seconds.",
    )
    loss_col, ap_col, fp_col, action_col = st.columns(4)
    loss_col.metric("Loss class", "Returns")
    ap_col.metric("Held-out avg. precision", f"{metrics['average_precision']:.3f}")
    fp_col.metric("False positives", metrics["confusion_matrix"]["false_positive"])
    action_col.metric("Permitted action", "Monitor only")

    st.markdown(
        f"""
        <div class="rr-judge-grid">
          <div class="rr-judge-card"><span>01 · Detector</span><b>Scores before shipment</b>
          <p>Checkout-time price, quantity, discount, category, delivery and payment data
          produce a return probability and local explanation.</p></div>
          <div class="rr-judge-card"><span>02 · Honest evidence</span>
          <b>Measured on future orders</b>
          <p>{metrics['orders']:,} chronologically held-out orders produced
          {metrics['precision']:.1%} precision and {metrics['recall']:.1%} recall at the
          frozen threshold.</p></div>
          <div class="rr-judge-card"><span>03 · Safety response</span>
          <b>Economics control action</b>
          <p>Estimated savings were ₹{metrics['estimated_savings_per_1000']:,.0f} per 1,000,
          so the release automatically remains in shadow mode with no customer blocking.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### How an order moves through the system")
    st.markdown(
        """
        <div class="rr-architecture">
          <div><b>01 · ORDER</b>Checkout details arrive</div>
          <div><b>02 · VALIDATE</b>Schema and ranges checked</div>
          <div><b>03 · PREPARE</b>Leakage-safe features only</div>
          <div><b>04 · SCORE</b>Frozen CatBoost probability</div>
          <div><b>05 · EXPLAIN</b>SHAP reason codes</div>
          <div><b>06 · GUARD</b>Monitor and collect outcome</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Return outcomes, return costs, identifiers, age, gender and location never enter scoring."
    )

    story_col, demo_col = st.columns([1.08, 0.92], gap="large")
    with story_col:
        with st.container(border=True):
            st.markdown("#### Why this is more than a trained model")
            st.markdown(
                "- Leakage prevention is enforced by an allowlist and tests\n"
                "- Model and threshold were frozen before the final test\n"
                "- False-positive friction is included in the cost policy\n"
                "- Batch drift, explanations and outcome monitoring are built in\n"
                "- Unsafe economics trigger a visible operational fallback"
            )
    with demo_col:
        with st.container(border=True):
            st.markdown("#### Recommended live walkthrough")
            st.markdown(
                "1. Open **Try the detector** and load the elevated preset\n"
                "2. Calculate the score and inspect its reason codes\n"
                "3. Run the one-click demo in **Batch risk review**\n"
                "4. Show **Policy Lab** false-positive economics\n"
                "5. Finish on **Model evidence** and the shadow-mode decision"
            )
            st.info(
                "Core message: the system detects risk, measures harm, "
                "and refuses unsafe action."
            )

with detector_tab:
    st.markdown(
        '<div class="rr-section-head"><h2>Score an order</h2>'
        "<p>Only information available before fulfilment is used. Post-return fields and "
        "customer identity are excluded.</p></div>",
        unsafe_allow_html=True,
    )
    feature_hint(
        "blue",
        "02",
        "Purpose of the detector",
        "Estimate return probability before shipment using checkout-time information only.",
    )
    for input_key, default_value in DEFAULT_ORDER_INPUTS.items():
        if input_key not in st.session_state:
            st.session_state[input_key] = default_value
    with st.container(border=False):
        st.markdown(
            '<div class="rr-preset-row"><b>Demo presets</b><br>'
            '<span style="color:#64748b;font-size:.75rem">Load a reproducible scenario, '
            "then calculate its return risk.</span></div>",
            unsafe_allow_html=True,
        )
        routine_col, elevated_col, warning_col = st.columns(3)
        routine_col.button(
            "Load routine order",
            on_click=apply_demo_preset,
            args=("routine",),
            use_container_width=True,
        )
        elevated_col.button(
            "Load elevated-risk order",
            on_click=apply_demo_preset,
            args=("elevated",),
            use_container_width=True,
        )
        warning_col.button(
            "Load reliability-warning order",
            on_click=apply_demo_preset,
            args=("warning",),
            use_container_width=True,
        )
        if st.session_state.get("demo_preset_notice"):
            st.caption(st.session_state["demo_preset_notice"])
    input_col, result_col = st.columns([1.12, 0.88], gap="large")
    with input_col:
        with st.container(border=True):
            st.markdown(
                '<div class="rr-panel-head"><span class="rr-panel-icon">▣</span>'
                "<b>Order details</b></div><div class=\"rr-panel-caption\">Enter the "
                "checkout information available to the merchant.</div>",
                unsafe_allow_html=True,
            )
            with st.form("order_form", border=False):
                first, second = st.columns(2)
                with first:
                    product_category = st.selectbox(
                        "Product category",
                        ["Clothing", "Electronics", "Books", "Home Appliances", "Toys"],
                        key="order_product_category",
                        help="Captures category-specific return patterns in the development data.",
                    )
                    order_quantity = st.number_input(
                        "Quantity",
                        min_value=1,
                        max_value=100,
                        step=1,
                        key="order_quantity",
                        help="Number of units included in this order.",
                    )
                    shipping_method = st.selectbox(
                        "Shipping method",
                        ["Standard", "Express", "Next-Day"],
                        key="order_shipping",
                        help="Delivery option selected before fulfilment.",
                    )
                    order_date = st.date_input(
                        "Order date",
                        key="order_date",
                        help="Used for seasonal patterns and out-of-period reliability warnings.",
                    )
                with second:
                    product_price = st.number_input(
                        "Product price (₹)",
                        min_value=1.0,
                        key="order_product_price",
                        help="Listed price per unit before the checkout discount.",
                    )
                    discount_applied = st.slider(
                        "Discount (%)",
                        min_value=0.0,
                        max_value=100.0,
                        step=0.5,
                        key="order_discount",
                        help="Percentage price reduction applied at checkout.",
                    )
                    payment_method = st.selectbox(
                        "Payment method",
                        ["Credit Card", "Debit Card", "Wallet", "COD"],
                        key="order_payment",
                        help="Payment method chosen at checkout; no card or account data is used.",
                    )
                    computed_value = product_price * order_quantity * (
                        1 - discount_applied / 100
                    )
                    st.metric("Calculated order value", f"₹{computed_value:,.2f}")
                submitted = st.form_submit_button(
                    "Calculate return risk  →", type="primary"
                )

    result = None
    prediction_id = None
    if submitted:
        order = OrderRequest(
            product_category=product_category,
            product_price=product_price,
            order_quantity=order_quantity,
            discount_applied=discount_applied,
            shipping_method=shipping_method,
            payment_method=payment_method,
            order_date=order_date,
        )
        result = scorer.score(order)
        try:
            prediction_id = record_score(monitoring_store, result, "dashboard_single")
        except sqlite3.Error as error:
            st.error(f"Risk calculated, but the audit record could not be saved: {error}")
            st.stop()

    with result_col:
        with st.container(border=True):
            st.markdown(
                '<div class="rr-panel-head"><span class="rr-panel-icon">◉</span>'
                "<b>Risk analysis</b></div>",
                unsafe_allow_html=True,
            )
            if result is None:
                st.markdown(
                    """
                    <div class="rr-empty">
                      <div class="rr-empty-orb">⌁</div>
                      <h3>Your decision panel will appear here</h3>
                      <p>Submit an order to see its return probability, threshold check,
                      explanation and permitted merchant action.</p>
                      <div class="rr-mini-flow"><span>Risk score</span><span>Top factors</span>
                      <span>Safe action</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                gauge_value = min(max(result.risk_score * 100, 0), 100)
                input_warnings = [
                    warning
                    for warning in result.warnings
                    if not warning.startswith("Held-out testing")
                ]
                verdict_class = (
                    "rr-verdict-high"
                    if result.would_flag_under_frozen_policy
                    else "rr-verdict-low"
                )
                verdict_text = (
                    "ABOVE REVIEW LINE"
                    if result.would_flag_under_frozen_policy
                    else "BELOW REVIEW LINE"
                )
                if input_warnings:
                    recommendation = "Abstain — reliability warning"
                    recommendation_tone = "warning"
                elif result.would_flag_under_frozen_policy:
                    recommendation = "Human-review candidate"
                    recommendation_tone = "review"
                else:
                    recommendation = "No review signal"
                    recommendation_tone = "routine"
                st.markdown(
                    f"""
                    <div class="rr-result">
                      <div class="rr-score-zone">
                        <div class="rr-gauge" style="background:conic-gradient(
                        #2e67ef 0 {gauge_value:.1f}%,#e8edf5 {gauge_value:.1f}% 100%);">
                          <div class="rr-score"><strong>{result.risk_score:.1%}</strong>
                          <span>return risk</span></div>
                        </div>
                        <div class="rr-result-copy">
                          <span class="rr-verdict {verdict_class}">{verdict_text}</span>
                          <h3>{recommendation}</h3>
                          <p>Treat this prediction as a decision-support signal, not a verdict
                          about the customer.</p>
                        </div>
                      </div>
                      <div class="rr-fact-list">
                        <div class="rr-fact"><span>Decision threshold</span>
                        <b>{result.decision_threshold:.1%}</b></div>
                        <div class="rr-fact"><span>Model release</span>
                        <b>{result.release_id.replace('return-risk-', '')[:8]} verified</b></div>
                        <div class="rr-fact"><span>Prediction stage</span>
                        <b>Pre-shipment</b></div>
                      </div>
                      <div class="decision"><span>◉</span><span><b>Safe action: monitor only.</b>
                      Save the result for outcome feedback. Do not block the order or change
                      the customer's return rights.</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                for warning in input_warnings:
                    st.warning(warning)
                st.markdown(
                    f"""
                    <div class="rr-action-grid">
                      <div class="rr-action-card rr-action-{recommendation_tone}">
                        <span>Decision support</span>
                        <strong>{html.escape(recommendation)}</strong>
                        <small>Order-level model recommendation</small>
                      </div>
                      <div class="rr-action-card rr-action-operational">
                        <span>Permitted operational action</span>
                        <strong>Monitor only</strong>
                        <small>Release-wide safety policy</small>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                reasons = pd.DataFrame(
                    [reason.model_dump() for reason in result.reasons]
                ).head(3)
                reasons["feature"] = reasons["feature"].map(readable_feature)
                reasons["effect"] = reasons["direction"].map(
                    {"raised": "Raised risk", "lowered": "Lowered risk"}
                )
                with st.expander("Why this score?", expanded=True):
                    threshold_position = (
                        "above" if result.would_flag_under_frozen_policy else "below"
                    )
                    st.markdown(
                        f"""
                        <div class="rr-score-explain-summary">
                        <b>This order scored {result.risk_score:.1%}.</b> The review line is
                        {result.decision_threshold:.1%}, so the score is {threshold_position}
                        the review line. The permitted action remains monitor only.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.caption("These three factors influenced this order's score the most.")
                    reason_cards = []
                    for row in reasons.to_dict(orient="records"):
                        plain_explanation = plain_reason(
                            str(row["reason"]), str(row["direction"])
                        )
                        reason_cards.append(
                            '<div class="rr-reason-card">'
                            '<div class="rr-reason-head">'
                            f"<strong>{html.escape(str(row['feature']))}</strong>"
                            f'<span class="rr-effect-{row["direction"]}">'
                            f"{html.escape(str(row['effect']))}</span></div>"
                            f'<div class="rr-reason-value">This order: '
                            f"{html.escape(str(row['value']))}</div>"
                            f"<p>{html.escape(plain_explanation)}</p></div>"
                        )
                    st.markdown(
                        '<div class="rr-reason-list">' + "".join(reason_cards) + "</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "Past return rates explain the pattern the model learned. They do not "
                        "mean that any factor causes a return."
                    )
                st.caption(
                    f"Audit ID `{prediction_id}` · record the outcome after the return window"
                )
                with st.expander("Technical details (sample sizes and SHAP)"):
                    st.caption("Detailed development-data evidence used for the explanations:")
                    for row in reasons.to_dict(orient="records"):
                        st.markdown(
                            f"**{row['feature']}:** {row['reason']}",
                        )
                    st.caption("SHAP contributions are shown in raw log-odds space:")
                    st.dataframe(
                        reasons[["feature", "shap_log_odds"]],
                        column_config={
                            "feature": "Feature",
                            "shap_log_odds": st.column_config.NumberColumn(
                                "SHAP contribution", format="%.3f"
                            ),
                        },
                        hide_index=True,
                        width="stretch",
                    )

    st.markdown(
        """
        <div class="rr-trust-strip">
          <div class="rr-trust-item"><b>Checkout-time only</b>
          <span>Post-return fields excluded</span></div>
          <div class="rr-trust-item"><b>Cost-aware threshold</b>
          <span>False positives are counted</span></div>
          <div class="rr-trust-item"><b>Outcome feedback</b>
          <span>Every prediction is auditable</span></div>
          <div class="rr-trust-item"><b>Defense-only</b>
          <span>No automatic rejection</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with monitoring_tab:
    st.subheader("Shadow outcomes")
    st.caption("Close the loop without storing customer attributes or changing the frozen model.")
    feature_hint(
        "mint",
        "04",
        "Purpose of outcome monitoring",
        "Record completed returns and measure performance on new orders without intervening.",
    )
    if st.session_state.pop("outcome_saved", False):
        st.success("Outcome saved. Live metrics are updated.")

    summary = monitoring_store.summary()
    recent = monitoring_store.recent_predictions(limit=100)
    total_col, complete_col, pending_col, action_col = st.columns(4)
    total_col.metric("Predictions", summary["total_predictions"])
    complete_col.metric("Outcomes received", summary["completed_outcomes"])
    pending_col.metric("Outcomes pending", summary["pending_outcomes"])
    action_col.metric("Customer interventions", 0)

    pending = [row for row in recent if row["returned"] is None]
    if pending:
        with st.expander("Record a completed outcome", expanded=True):
            with st.form("outcome_form"):
                selected_id = st.selectbox(
                    "Prediction",
                    [row["prediction_id"] for row in pending],
                    format_func=lambda value: next(
                        (
                            f"{value[:8]}… · risk {row['risk_score']:.1%}"
                            for row in pending
                            if row["prediction_id"] == value
                        ),
                        value,
                    ),
                )
                outcome_label = st.radio(
                    "Observed outcome", ["Returned", "Not returned"], horizontal=True
                )
                observed_cost = st.number_input(
                    "Return cost (optional, ₹)", min_value=0.0, value=0.0
                )
                outcome_submitted = st.form_submit_button("Save outcome", type="primary")
            if outcome_submitted:
                try:
                    monitoring_store.record_outcome(
                        selected_id,
                        returned=outcome_label == "Returned",
                        observed_return_cost=observed_cost if observed_cost > 0 else None,
                    )
                except PredictionNotFoundError as error:
                    st.error(str(error))
                except OutcomeAlreadyRecordedError as error:
                    st.warning(str(error))
                else:
                    st.session_state["outcome_saved"] = True
                    st.rerun()
    elif not recent:
        st.info("Score an order first. It will appear here for later outcome feedback.")

    if summary["completed_outcomes"]:
        st.markdown("### Live metrics on completed outcomes")
        precision_col, recall_col, f1_col, return_col = st.columns(4)
        precision_col.metric(
            "Precision", "N/A" if summary["precision"] is None else f"{summary['precision']:.1%}"
        )
        recall_col.metric(
            "Recall", "N/A" if summary["recall"] is None else f"{summary['recall']:.1%}"
        )
        f1_col.metric("F1", "N/A" if summary["f1"] is None else f"{summary['f1']:.1%}")
        return_col.metric("Return rate", f"{summary['observed_return_rate']:.1%}")
        policy = summary["counterfactual_frozen_policy"]
        st.caption(
            "Counterfactual frozen-policy savings: "
            f"₹{policy['savings_per_1000_orders']:,.0f} per 1,000 completed outcomes. "
            "Actual interventions remain zero."
        )

    if recent:
        with st.expander("Recent audit records"):
            recent_frame = pd.DataFrame(recent)
            recent_frame["prediction_id"] = (
                recent_frame["prediction_id"].str.slice(0, 8) + "…"
            )
            recent_frame["outcome"] = recent_frame["returned"].map(
                {True: "Returned", False: "Not returned"}
            ).fillna("Pending")
            st.dataframe(
                recent_frame[
                    ["prediction_id", "source", "risk_score", "would_flag", "outcome"]
                ],
                hide_index=True,
                width="stretch",
            )

    def render_batch_review() -> None:
        st.markdown(
            '<div class="evidence-note"><b>Estimate financial exposure—not return '
            "probability.</b> These merchant assumptions never change the model's risk "
            "score. They are used only to estimate and rank the possible financial loss "
            "from each return.</div>",
            unsafe_allow_html=True,
        )
        st.write("")
        exposure_first, exposure_second = st.columns(2)
        with exposure_first:
            with st.container(border=True):
                reverse_logistics_cost = st.number_input(
                    "Estimated reverse-logistics cost (₹)",
                    min_value=0.0,
                    value=100.0,
                    step=25.0,
                    help=(
                        "Estimated shipping, handling and inspection cost for one returned "
                        "order. It is used only for financial prioritization and never "
                        "enters the model."
                    ),
                )
        with exposure_second:
            with st.container(border=True):
                value_loss_rate = st.slider(
                    "Estimated merchandise loss (%)",
                    min_value=0,
                    max_value=100,
                    value=25,
                    help=(
                        "Percentage of order value expected to be lost through damage, "
                        "markdown or failed resale. It does not change the return-risk score."
                    ),
                )
        st.markdown(
            f'<div class="rr-exposure-formula"><b>Cost calculation:</b> Estimated return '
            f"loss for each order = ₹{reverse_logistics_cost:,.0f} reverse-logistics cost + "
            f"{value_loss_rate}% of its order value. These values affect only financial "
            "exposure ranking.</div>",
            unsafe_allow_html=True,
        )
        sample = demo_batch_frame()
        demo_col, download_col = st.columns(2)
        run_demo = demo_col.button(
            "Run the 40-order reviewer demo",
            type="primary",
            use_container_width=True,
            help="Scores deterministic synthetic checkout orders for a reviewer walkthrough.",
        )
        download_col.download_button(
            "Download CSV template",
            sample.to_csv(index=False).encode("utf-8"),
            file_name="return_risk_batch_sample.csv",
            mime="text/csv",
            use_container_width=True,
        )
        uploaded = st.file_uploader(
            "Upload merchant order CSV",
            type=["csv"],
            help="Up to 1,000 checkout-time order rows in the supplied CSV format.",
        )
        if run_demo:
            uploaded_frame = sample
            st.caption(
                "Using the deliberately varied 40-order reviewer batch. An elevated drift "
                "warning is expected and demonstrates the safety guardrail."
            )
        elif uploaded is not None:
            uploaded_frame = pd.read_csv(uploaded)
        else:
            uploaded_frame = None

        if uploaded_frame is not None:
            st.dataframe(uploaded_frame.head(20), hide_index=True, width="stretch")
            score_requested = run_demo or st.button("Score uploaded batch", type="primary")
            if score_requested:
                try:
                    batch = scorer.score_batch(uploaded_frame)
                    logged_rows = []
                    for row in batch.results:
                        prediction_id = monitoring_store.record_prediction(
                            release_id=batch.release_id,
                            source=(
                                "dashboard_demo_batch" if run_demo else "dashboard_batch"
                            ),
                            risk_score=row.risk_score,
                            decision_threshold=row.decision_threshold,
                            would_flag=row.would_flag_under_frozen_policy,
                            computed_order_value=row.computed_order_value,
                        )
                        logged_rows.append(
                            row.model_copy(update={"prediction_id": prediction_id})
                        )
                    batch = batch.model_copy(update={"results": logged_rows})
                except (ValueError, sqlite3.Error) as error:
                    st.error(str(error))
                else:
                    scored = prepare_batch_review_frame(
                        batch.results,
                        uploaded_frame,
                        reverse_logistics_cost,
                        value_loss_rate,
                    )
                    st.success(
                        f"Scored {batch.scored_rows} orders · {batch.would_flag_count} above "
                        "threshold · 0 interventions"
                    )
                    scored_col, invalid_col, flagged_col, drift_col = st.columns(4)
                    scored_col.metric("Scored orders", batch.scored_rows)
                    invalid_col.metric("Invalid rows", batch.invalid_rows)
                    flagged_col.metric("Above threshold", batch.would_flag_count)
                    drift_col.metric(
                        "Drift status",
                        batch.drift.get("overall_severity", "unavailable").replace("_", " "),
                    )
                    drift_severity = batch.drift.get("overall_severity")
                    if drift_severity in {"warning", "high"}:
                        st.warning(
                            "Batch input drift is elevated. Scores remain shadow-only and "
                            "should not be used for customer-facing action."
                        )
                    elif drift_severity == "insufficient_data":
                        st.info(
                            "At least 30 valid orders are required for a reliable batch "
                            "drift decision."
                        )
                    if not scored.empty:
                        exposure_col, top_order_col = st.columns(2)
                        exposure_col.metric(
                            "Total expected-loss exposure",
                            f"₹{scored['expected_loss_exposure'].sum():,.0f}",
                        )
                        top_order_col.metric(
                            "Highest-priority order",
                            str(scored.iloc[0].get("order_reference") or "Row 1"),
                        )
                        risk_items, category_items = batch_distribution_items(scored)
                        st.markdown(
                            '<div class="rr-viz-grid">'
                            + horizontal_bar_chart("Risk-score distribution", risk_items)
                            + horizontal_bar_chart("Category distribution", category_items)
                            + "</div>",
                            unsafe_allow_html=True,
                        )
                    st.caption(
                        "Priority is risk probability × estimated return loss. It is a "
                        "counterfactual review aid and never triggers an intervention."
                    )
                    if batch.validation_errors:
                        with st.expander("Invalid row details"):
                            invalid_rows = pd.DataFrame(
                                [error.model_dump() for error in batch.validation_errors]
                            )
                            invalid_rows["errors"] = invalid_rows["errors"].map(
                                lambda errors: " | ".join(errors)
                            )
                            st.dataframe(
                                invalid_rows,
                                hide_index=True,
                                width="stretch",
                            )
                    st.dataframe(
                        scored,
                        column_config={
                            "risk_score": st.column_config.NumberColumn(
                                "Risk score", format="percent"
                            ),
                            "estimated_return_loss": st.column_config.NumberColumn(
                                "Estimated return loss", format="₹%.2f"
                            ),
                            "expected_loss_exposure": st.column_config.NumberColumn(
                                "Expected-loss exposure", format="₹%.2f"
                            ),
                            "priority_rank": "Priority",
                        },
                        hide_index=True,
                        width="stretch",
                    )
                    st.download_button(
                        "Download shadow scores",
                        scored.to_csv(index=False).encode("utf-8"),
                        file_name="return_risk_shadow_scores.csv",
                        mime="text/csv",
                    )
                    merchant_report = merchant_risk_report_html(
                        scored,
                        batch_summary={
                            "total_rows": batch.total_rows,
                            "scored_rows": batch.scored_rows,
                            "invalid_rows": batch.invalid_rows,
                            "would_flag_count": batch.would_flag_count,
                        },
                        drift=batch.drift,
                        release_id=batch.release_id,
                        reverse_logistics_cost=reverse_logistics_cost,
                        merchandise_loss_rate=value_loss_rate,
                        generated_at_utc=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
                    )
                    st.download_button(
                        "Download merchant risk report",
                        merchant_report.encode("utf-8"),
                        file_name="merchant_return_risk_report.html",
                        mime="text/html",
                    )


with batch_tab:
    st.subheader("Batch risk review")
    st.caption(
        "Score one built-in reviewer batch or upload up to 1,000 merchant orders. "
        "Every result remains monitoring-only."
    )
    feature_hint(
        "blue",
        "03",
        "Purpose of batch risk review",
        "Prioritize many orders by expected return-loss exposure and inspect input drift.",
    )
    st.markdown(
        """
        <div class="rr-batch-guide">
          <div class="rr-batch-step"><b>01 · SET COSTS</b>
          Enter the merchant's estimated cost of a return.</div>
          <div class="rr-batch-step"><b>02 · SCORE ORDERS</b>
          Run the reviewer demo or upload the CSV template.</div>
          <div class="rr-batch-step"><b>03 · REVIEW OUTPUT</b>
          Inspect ranked exposure, input drift and the merchant report.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_batch_review()

with policy_tab:
    st.subheader("Policy Lab")
    st.caption(
        "Explore how merchant costs change a validation-only review policy. "
        "This simulator cannot change the frozen model or shadow-mode safety lock."
    )
    feature_hint(
        "amber",
        "05",
        "Purpose of the policy simulator",
        "Test whether verification could be worthwhile under different merchant cost assumptions.",
    )
    st.markdown(
        '<div class="evidence-note"><b>Counterfactual simulation only.</b> '
        "Calculations use aggregate validation confusion counts. The held-out test is "
        "never used to select a new threshold.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    assumptions_col, explanation_col = st.columns([0.92, 1.08], gap="large")
    with assumptions_col:
        with st.container(border=True):
            st.markdown("#### Merchant assumptions")
            verification_cost = st.number_input(
                "Verification cost per reviewed order (₹)",
                min_value=0.0,
                value=25.0,
                step=5.0,
                key="policy_verification_cost",
            )
            false_positive_cost = st.number_input(
                "False-positive customer-friction cost (₹)",
                min_value=0.0,
                value=50.0,
                step=10.0,
                key="policy_false_positive_cost",
            )
            missed_return_cost = st.number_input(
                "Loss from a missed return (₹)",
                min_value=0.0,
                value=200.0,
                step=25.0,
                key="policy_missed_return_cost",
            )
            effectiveness = st.slider(
                "Verification effectiveness",
                min_value=0,
                max_value=100,
                value=50,
                key="policy_effectiveness",
                help="Assumed percentage of return loss prevented by a successful review.",
            )
            review_capacity = st.slider(
                "Maximum review capacity",
                min_value=1,
                max_value=100,
                value=30,
                key="policy_review_capacity",
                help="Maximum percentage of orders the merchant can review.",
            )

    assumptions = CostAssumptions(
        verification_cost=verification_cost,
        false_positive_friction_cost=false_positive_cost,
        missed_return_cost=missed_return_cost,
        intervention_effectiveness=effectiveness / 100,
    )
    simulation = evaluate_policy_frontier(
        policy_frontier["frontier"], assumptions, review_capacity / 100
    )
    selected_policy = simulation["selected"]

    with explanation_col:
        with st.container(border=True):
            st.markdown("#### Simulated validation policy")
            if selected_policy["orders_flagged"] == 0:
                st.warning(
                    "Under these assumptions, the lowest-cost validation policy is to "
                    "review no orders."
                )
            else:
                st.success(
                    "A lower-cost review policy exists on validation under these assumptions."
                )
            first, second, third = st.columns(3)
            first.metric("Suggested threshold", f"{selected_policy['threshold']:.1%}")
            second.metric("Review rate", f"{selected_policy['flagged_rate']:.1%}")
            third.metric(
                "Savings / 1,000",
                f"₹{selected_policy['savings_per_1000_orders']:,.0f}",
            )
            precision_col, recall_col, fp_col = st.columns(3)
            precision_col.metric("Precision", f"{selected_policy['precision']:.1%}")
            recall_col.metric("Recall", f"{selected_policy['recall']:.1%}")
            fp_col.metric("False positives", selected_policy["false_positive"])
            st.markdown(
                '<div class="decision"><span>◉</span><span><b>Operational decision remains '
                "monitor only.</b> Final held-out economics were negative, so this "
                "validation scenario cannot authorize customer-facing action.</span></div>",
                unsafe_allow_html=True,
            )

    curve = pd.DataFrame(simulation["curve"])
    curve = (
        curve.groupby("flagged_rate", as_index=False)["savings_per_1000_orders"]
        .max()
        .sort_values("flagged_rate")
    )
    curve["Review capacity used (%)"] = curve["flagged_rate"] * 100
    curve = curve.rename(columns={"savings_per_1000_orders": "Savings per 1,000 (₹)"})
    st.markdown("#### Interactive validation cost frontier")
    st.caption(
        "Changes when you adjust the five Policy Lab assumptions above. The underlying "
        "validation order rankings and confusion counts remain frozen; single-order, batch, "
        "held-out test, and live monitoring data do not alter this curve."
    )
    st.markdown(
        policy_frontier_chart(curve),
        unsafe_allow_html=True,
    )
    st.caption(
        "Above-zero values are hypothetical validation estimates. Intervention effectiveness "
        "has not been measured on real merchant orders."
    )

    st.markdown("#### Champion versus candidates")
    comparison = pd.DataFrame(model_selection["models"])
    comparison = comparison.rename(
        columns={
            "model": "Model",
            "status": "Decision",
            "roc_auc": "ROC-AUC",
            "average_precision": "Average precision",
            "brier_score": "Brier score",
            "top_10_percent_precision": "Top-10% precision",
        }
    )
    st.dataframe(
        comparison,
        column_config={
            "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
            "Average precision": st.column_config.NumberColumn(format="%.3f"),
            "Brier score": st.column_config.NumberColumn(format="%.3f"),
            "Top-10% precision": st.column_config.NumberColumn(format="percent"),
        },
        hide_index=True,
        width="stretch",
    )
    st.caption(model_selection["decision"])

with evidence_tab:
    st.subheader("Final held-out evidence")
    feature_hint(
        "slate",
        "06",
        "Purpose of model evidence",
        "Inspect frozen test results, false-positive cost, leakage controls, and limitations.",
    )
    st.markdown(
        '<div class="evidence-note"><b>Modest ranking signal; unsafe intervention '
        "economics.</b> The system correctly falls back to shadow monitoring.</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    precision_col, recall_col, fp_col, savings_col = st.columns(4)
    precision_col.metric("Precision", f"{metrics['precision']:.1%}")
    recall_col.metric("Recall", f"{metrics['recall']:.1%}")
    fp_col.metric("False positives", metrics["confusion_matrix"]["false_positive"])
    savings_col.metric("Savings / 1,000", f"₹{metrics['estimated_savings_per_1000']:,.0f}")
    st.caption(
        f"Chronological test · {metrics['orders']:,} orders · opened once after release freeze"
    )

    allowed, blocked = st.columns(2)
    with allowed:
        st.markdown(
            '<div class="allow"><b>✓ Prediction-time inputs</b><br>Price, quantity, discount, '
            "order value, date parts, category, shipping method and payment method.</div>",
            unsafe_allow_html=True,
        )
    with blocked:
        st.markdown(
            '<div class="block"><b>✕ Excluded from scoring</b><br>Return outcomes and costs, '
            "identifiers, age, gender, location and sustainability outcomes.</div>",
            unsafe_allow_html=True,
        )

    with st.expander("Full model metrics and cost interval"):
        first, second, third = st.columns(3)
        first.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        second.metric("Average precision", f"{metrics['average_precision']:.3f}")
        third.metric("F1", f"{metrics['f1']:.3f}")
        interval = metrics["savings_per_1000_interval"]
        st.write(
            "Estimated savings are negative across the 95% bootstrap interval: "
            f"₹{interval['lower']:,.0f} to ₹{interval['upper']:,.0f} per 1,000 orders."
        )
        st.write(
            f"All frozen-threshold flags were {metrics['largest_flagged_group']} orders "
            f"({metrics['largest_group_share']:.0%} concentration)."
        )

    with st.expander("Evaluation controls and limitations"):
        st.markdown(
            "- Chronological train, validation and held-out test split\n"
            "- Model and threshold frozen before opening the test set\n"
            "- Startup verifies the frozen model hash\n"
            "- SHAP describes model behavior, not causality\n"
            "- Synthetic source data; real merchant outcomes are required before deployment"
        )
        st.caption(f"Release {card['release_id']} · actual action monitor_only")
