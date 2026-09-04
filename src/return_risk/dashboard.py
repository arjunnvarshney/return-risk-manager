from __future__ import annotations

import html
from collections.abc import Iterable
from datetime import date, timedelta

import pandas as pd

from return_risk.policy import expected_loss_exposure

RISK_BIN_EDGES = [0, 0.2, 0.4, 0.6, 0.8, 1.000001]
RISK_BIN_LABELS = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]


def demo_batch_frame(order_count: int = 40) -> pd.DataFrame:
    """Return a deterministic, schema-valid batch for a reviewer walkthrough."""
    if not 1 <= order_count <= 1_000:
        raise ValueError("order_count must be between 1 and 1,000.")
    categories = ["Clothing", "Electronics", "Books", "Home Appliances", "Toys"]
    base_prices = {
        "Clothing": 1200.0,
        "Electronics": 4800.0,
        "Books": 450.0,
        "Home Appliances": 3500.0,
        "Toys": 900.0,
    }
    discounts = [5.0, 20.0, 35.0, 50.0]
    shipping_methods = ["Standard", "Express", "Next-Day"]
    payment_methods = ["Credit Card", "Debit Card", "Wallet", "COD"]
    start_date = date(2025, 7, 1)
    rows = []
    for index in range(order_count):
        category = categories[index % len(categories)]
        rows.append(
            {
                "order_reference": f"DEMO-{index + 1:03d}",
                "product_category": category,
                "product_price": base_prices[category] + (index % 3) * 100,
                "order_quantity": index % 3 + 1,
                "discount_applied": discounts[index % len(discounts)],
                "shipping_method": shipping_methods[index % len(shipping_methods)],
                "payment_method": payment_methods[index % len(payment_methods)],
                "order_date": (start_date + timedelta(days=index)).isoformat(),
            }
        )
    return pd.DataFrame(rows)


def prepare_batch_review_frame(
    results: Iterable,
    input_frame: pd.DataFrame,
    reverse_logistics_cost: float,
    merchandise_loss_rate: float,
) -> pd.DataFrame:
    if reverse_logistics_cost < 0:
        raise ValueError("reverse_logistics_cost cannot be negative.")
    if not 0 <= merchandise_loss_rate <= 100:
        raise ValueError("merchandise_loss_rate must be between 0 and 100.")

    records = [
        result.model_dump() if hasattr(result, "model_dump") else dict(result)
        for result in results
    ]
    scored = pd.DataFrame(records)
    if scored.empty:
        return scored

    scored["estimated_return_loss"] = (
        reverse_logistics_cost
        + scored["computed_order_value"] * merchandise_loss_rate / 100
    ).round(2)
    scored["expected_loss_exposure"] = scored.apply(
        lambda row: expected_loss_exposure(
            row["risk_score"], row["estimated_return_loss"]
        ),
        axis=1,
    )
    scored["priority_rank"] = (
        scored["expected_loss_exposure"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    category_by_row = {
        position: str(row.get("product_category", "Unknown"))
        for position, (_, row) in enumerate(input_frame.iterrows(), start=2)
    }
    scored["product_category"] = scored["row_number"].map(category_by_row)
    return scored.sort_values(["priority_rank", "row_number"], kind="stable")


def batch_distribution_items(
    scored: pd.DataFrame,
) -> tuple[list[tuple[str, int, str]], list[tuple[str, int, str]]]:
    if scored.empty:
        return [], []
    risk_bands = pd.cut(
        scored["risk_score"],
        bins=RISK_BIN_EDGES,
        labels=RISK_BIN_LABELS,
        include_lowest=True,
        right=False,
    )
    risk_counts = risk_bands.value_counts(sort=False)
    risk_items = [
        (str(label), int(value), f"{int(value)} orders")
        for label, value in risk_counts.items()
    ]
    category_summary = (
        scored.groupby("product_category", dropna=False)
        .agg(
            orders=("row_number", "count"),
            flagged=("would_flag_under_frozen_policy", "sum"),
        )
        .sort_values("orders", ascending=False)
    )
    category_items = [
        (
            str(category),
            int(row["orders"]),
            f"{int(row['orders'])} · {int(row['flagged'])} flagged",
        )
        for category, row in category_summary.iterrows()
    ]
    return risk_items, category_items


def _report_table(headers: list[str], rows: list[list[object]]) -> str:
    header_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    row_html = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
        + "</tr>"
        for row in rows
    )
    if not rows:
        row_html = f'<tr><td colspan="{len(headers)}">No scored orders</td></tr>'
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{row_html}</tbody></table>"


def merchant_risk_report_html(
    scored: pd.DataFrame,
    *,
    batch_summary: dict,
    drift: dict,
    release_id: str,
    reverse_logistics_cost: float,
    merchandise_loss_rate: float,
    generated_at_utc: str,
) -> str:
    """Return a portable, escaped shadow-mode report for one scored batch."""
    if reverse_logistics_cost < 0:
        raise ValueError("reverse_logistics_cost cannot be negative.")
    if not 0 <= merchandise_loss_rate <= 100:
        raise ValueError("merchandise_loss_rate must be between 0 and 100.")

    expected_exposure = (
        float(scored["expected_loss_exposure"].sum()) if not scored.empty else 0.0
    )
    mean_risk = float(scored["risk_score"].mean()) if not scored.empty else 0.0
    highest_risk = float(scored["risk_score"].max()) if not scored.empty else 0.0
    threshold = (
        float(scored["decision_threshold"].iloc[0])
        if not scored.empty and "decision_threshold" in scored
        else 0.0
    )

    risk_items, category_items = batch_distribution_items(scored)
    risk_table = _report_table(
        ["Risk band", "Orders"],
        [[label, value] for label, value, _ in risk_items],
    )
    category_table = _report_table(
        ["Category", "Summary"],
        [[label, detail] for label, _, detail in category_items],
    )

    priority_rows = []
    if not scored.empty:
        for _, row in scored.nsmallest(10, "priority_rank").iterrows():
            reference = row.get("order_reference")
            if pd.isna(reference) or not str(reference).strip():
                reference = f"CSV row {int(row['row_number'])}"
            priority_rows.append(
                [
                    int(row["priority_rank"]),
                    reference,
                    row.get("product_category", "Unknown"),
                    f"{float(row['risk_score']):.1%}",
                    f"₹{float(row['expected_loss_exposure']):,.2f}",
                    "Would review" if row["would_flag_under_frozen_policy"] else "Monitor",
                ]
            )
    priority_table = _report_table(
        ["Priority", "Order", "Category", "Risk", "Exposure", "Frozen policy"],
        priority_rows,
    )

    drift_features = sorted(
        drift.get("features", {}).items(),
        key=lambda item: float(item[1].get("psi", 0)),
        reverse=True,
    )[:5]
    drift_table = _report_table(
        ["Feature", "PSI", "Status"],
        [
            [feature.replace("_", " ").title(), f"{details['psi']:.3f}", details["severity"]]
            for feature, details in drift_features
        ],
    )
    drift_status = drift.get("overall_severity", drift.get("status", "unavailable"))
    drift_status = str(drift_status).replace("_", " ").title()

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Merchant Return-Risk Report</title>
  <style>
    :root{{--ink:#10213e;--muted:#64748b;--line:#dfe6f0;--blue:#245de1;--violet:#513592}}
    *{{box-sizing:border-box}} body{{margin:0;background:#f5f7fb;color:var(--ink);
    font:14px/1.5 Inter,Segoe UI,Arial,sans-serif}} main{{max-width:1050px;margin:28px auto;
    padding:0 22px 40px}} header{{padding:28px 30px;border-radius:18px;color:white;
    background:linear-gradient(125deg,#0b1d42,#2056bd)}} h1{{margin:4px 0 7px;font-size:30px}}
    h2{{margin:28px 0 10px;font-size:19px}} h3{{margin:0 0 10px;font-size:15px}}
    .eyebrow{{color:#85efe0;font-size:11px;font-weight:700;letter-spacing:.1em}}
    .muted{{color:var(--muted)}} .notice{{margin:16px 0;padding:13px 15px;border:1px solid
    #b9a6f3;border-radius:11px;background:#f3efff;color:var(--violet)}} .grid{{display:grid;
    grid-template-columns:repeat(4,1fr);gap:10px;margin:16px 0}} .card,.panel{{padding:16px;
    border:1px solid var(--line);border-radius:12px;background:white}} .card span{{display:block;
    color:var(--muted);font-size:11px}} .card b{{display:block;margin-top:5px;font-size:22px}}
    .two{{display:grid;grid-template-columns:1fr 1fr;gap:12px}} table{{width:100%;
    border-collapse:collapse;background:white}} th,td{{padding:9px 10px;border-bottom:1px solid
    var(--line);text-align:left;font-size:12px}} th{{color:#52627a;background:#f8faff}}
    .meta{{margin-top:22px;padding-top:14px;border-top:1px solid var(--line);font-size:11px;
    color:var(--muted)}} @media(max-width:700px){{.grid,.two{{grid-template-columns:1fr 1fr}}}}
    @media print{{body{{background:white}} main{{margin:0;max-width:none}}}}
  </style>
</head>
<body><main>
  <header><div class="eyebrow">RETURN RISK MANAGER · SHADOW REPORT</div>
  <h1>Merchant return-risk report</h1>
  <div>Pre-shipment risk prioritization with no customer-facing intervention</div></header>
  <div class="notice"><b>Monitor only.</b> This report must not be used to automatically
  reject orders, add customer friction, or restrict return rights.</div>
  <section class="grid">
    <div class="card"><span>Scored orders</span><b>{int(batch_summary['scored_rows']):,}</b></div>
    <div class="card"><span>Invalid rows</span><b>{int(batch_summary['invalid_rows']):,}</b></div>
    <div class="card"><span>Above frozen threshold</span>
    <b>{int(batch_summary['would_flag_count']):,}</b></div>
    <div class="card"><span>Expected-loss exposure</span><b>₹{expected_exposure:,.0f}</b></div>
  </section>
  <section class="two">
    <div class="panel"><h3>Risk snapshot</h3>
    <div>Mean risk: <b>{mean_risk:.1%}</b></div>
    <div>Highest risk: <b>{highest_risk:.1%}</b></div>
    <div>Frozen threshold: <b>{threshold:.1%}</b></div></div>
    <div class="panel"><h3>Merchant assumptions</h3>
    <div>Reverse-logistics cost: <b>₹{reverse_logistics_cost:,.2f}</b></div>
    <div>Merchandise loss: <b>{merchandise_loss_rate:.0f}% of order value</b></div>
    <div>Drift status: <b>{html.escape(drift_status)}</b></div></div>
  </section>
  <h2>Prioritized shadow review queue</h2>{priority_table}
  <h2>Batch composition</h2><section class="two">
    <div class="panel"><h3>Risk distribution</h3>{risk_table}</div>
    <div class="panel"><h3>Category distribution</h3>{category_table}</div>
  </section>
  <h2>Largest drift indicators</h2>{drift_table}
  <div class="meta">Generated {html.escape(generated_at_utc)} · Model release
  {html.escape(release_id)} · Total uploaded rows {int(batch_summary['total_rows']):,} ·
  Actual interventions 0 · Expected-loss exposure is a merchant-assumption scenario,
  not realized savings.</div>
</main></body></html>"""
