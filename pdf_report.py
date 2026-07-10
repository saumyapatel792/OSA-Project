"""
OSA-CDSS  ·  PDF Report Generator
===================================
Generates professional clinical PDF reports using ReportLab.
Three report types:
  1. Patient Clinical Report  — individual patient summary
  2. Population Report        — full cohort analytics
  3. Risk Assessment Report   — post-prediction summary
"""

import io
import datetime
import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics import renderPDF

# ── BRAND COLOURS ─────────────────────────────────────────
NAVY      = colors.HexColor("#1B2A4A")
BLUE      = colors.HexColor("#2A6496")
LIGHT_BLUE= colors.HexColor("#38BDF8")
GREEN     = colors.HexColor("#16A34A")
AMBER     = colors.HexColor("#D97706")
RED       = colors.HexColor("#DC2626")
PURPLE    = colors.HexColor("#7C3AED")
GREY_DARK = colors.HexColor("#4A5568")
GREY_MID  = colors.HexColor("#718096")
GREY_LIGHT= colors.HexColor("#E2E8F0")
WHITE     = colors.white
BG_LIGHT  = colors.HexColor("#F0F4F8")

RISK_COLORS = {
    "Low":      colors.HexColor("#16A34A"),
    "Moderate": colors.HexColor("#D97706"),
    "High":     colors.HexColor("#DC2626"),
    "Severe":   colors.HexColor("#7C3AED"),
}

# ── STYLES ────────────────────────────────────────────────
def get_styles():
    styles = getSampleStyleSheet()
    custom = {
        "ReportTitle": ParagraphStyle("ReportTitle",
            fontSize=20, textColor=WHITE, fontName="Helvetica-Bold",
            alignment=TA_LEFT, spaceAfter=4),
        "ReportSubtitle": ParagraphStyle("ReportSubtitle",
            fontSize=10, textColor=colors.HexColor("#93C5FD"),
            fontName="Helvetica", alignment=TA_LEFT, spaceAfter=2),
        "SectionTitle": ParagraphStyle("SectionTitle",
            fontSize=9, textColor=GREY_MID, fontName="Helvetica-Bold",
            alignment=TA_LEFT, spaceBefore=14, spaceAfter=6,
            textTransform="uppercase", letterSpacing=1.5),
        "BodyText": ParagraphStyle("BodyText",
            fontSize=10, textColor=GREY_DARK, fontName="Helvetica",
            alignment=TA_LEFT, spaceAfter=4, leading=15),
        "SmallText": ParagraphStyle("SmallText",
            fontSize=8, textColor=GREY_MID, fontName="Helvetica",
            alignment=TA_LEFT, spaceAfter=2),
        "MetricValue": ParagraphStyle("MetricValue",
            fontSize=22, fontName="Helvetica-Bold",
            alignment=TA_CENTER, spaceAfter=2),
        "MetricLabel": ParagraphStyle("MetricLabel",
            fontSize=8, textColor=GREY_MID, fontName="Helvetica-Bold",
            alignment=TA_CENTER, letterSpacing=1),
        "RiskBadge": ParagraphStyle("RiskBadge",
            fontSize=13, fontName="Helvetica-Bold",
            alignment=TA_CENTER),
        "TableHeader": ParagraphStyle("TableHeader",
            fontSize=8, textColor=WHITE, fontName="Helvetica-Bold",
            alignment=TA_CENTER),
        "TableCell": ParagraphStyle("TableCell",
            fontSize=8, textColor=GREY_DARK, fontName="Helvetica",
            alignment=TA_CENTER),
        "Footer": ParagraphStyle("Footer",
            fontSize=7, textColor=GREY_MID, fontName="Helvetica",
            alignment=TA_CENTER),
        "Disclaimer": ParagraphStyle("Disclaimer",
            fontSize=8, textColor=AMBER, fontName="Helvetica-Bold",
            alignment=TA_CENTER, spaceAfter=4),
    }
    return custom

# ── HEADER BANNER ─────────────────────────────────────────
def build_header_banner(title, subtitle, width=A4[0] - 40*mm):
    d = Drawing(width, 60)
    # Background
    d.add(Rect(0, 0, width, 60, fillColor=NAVY, strokeColor=LIGHT_BLUE, strokeWidth=2))
    # Left accent bar
    d.add(Rect(0, 0, 5, 60, fillColor=LIGHT_BLUE, strokeColor=None))
    # Title
    d.add(String(16, 38, title,
        fontName="Helvetica-Bold", fontSize=16, fillColor=colors.white))
    # Subtitle
    d.add(String(16, 22, subtitle,
        fontName="Helvetica", fontSize=9, fillColor=colors.HexColor("#93C5FD")))
    # Institution (right side)
    d.add(String(width - 10, 42, "Alliance University Hospital",
        fontName="Helvetica-Bold", fontSize=9, fillColor=colors.white,
        textAnchor="end"))
    d.add(String(width - 10, 28, "Dept. of Pulmonology & Sleep Medicine",
        fontName="Helvetica", fontSize=7, fillColor=colors.HexColor("#64748B"),
        textAnchor="end"))
    d.add(String(width - 10, 16, f"Generated: {datetime.datetime.now().strftime('%d %b %Y  %H:%M')}",
        fontName="Helvetica", fontSize=7, fillColor=colors.HexColor("#475569"),
        textAnchor="end"))
    return d

# ── METRIC BOX ROW ────────────────────────────────────────
def metric_box(label, value, color=BLUE, width=40*mm, height=22*mm):
    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height,
        fillColor=colors.HexColor("#F8FAFC"),
        strokeColor=GREY_LIGHT, strokeWidth=0.5))
    # Top colour bar
    d.add(Rect(0, height-4, width, 4, fillColor=color, strokeColor=None))
    d.add(String(width/2, height/2 + 2, str(value),
        fontName="Helvetica-Bold", fontSize=14, fillColor=color,
        textAnchor="middle"))
    d.add(String(width/2, 5, label.upper(),
        fontName="Helvetica-Bold", fontSize=6, fillColor=GREY_MID,
        textAnchor="middle"))
    return d

# ── RISK BADGE ────────────────────────────────────────────
def risk_badge_drawing(risk, score_pct, width=120, height=50):
    clr = RISK_COLORS.get(risk, BLUE)
    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height,
        fillColor=colors.HexColor("#F8FAFC"),
        strokeColor=clr, strokeWidth=1.5, rx=6, ry=6))
    d.add(String(width/2, height-16, risk.upper(),
        fontName="Helvetica-Bold", fontSize=13, fillColor=clr,
        textAnchor="middle"))
    d.add(String(width/2, 10, f"{score_pct:.1f}% OSA Probability",
        fontName="Helvetica", fontSize=8, fillColor=GREY_MID,
        textAnchor="middle"))
    return d

# ── PIE CHART ─────────────────────────────────────────────
def build_pie_chart(risk_counts, width=160, height=130):
    d = Drawing(width, height)
    pie = Pie()
    pie.x = 30; pie.y = 15
    pie.width = 90; pie.height = 90
    labels = list(risk_counts.keys())
    values = list(risk_counts.values())
    pie.data = values
    pie.labels = [f"{l} ({v})" for l, v in zip(labels, values)]
    pie.slices.strokeWidth = 1
    pie.slices.strokeColor = WHITE
    for i, lbl in enumerate(labels):
        pie.slices[i].fillColor = RISK_COLORS.get(lbl, BLUE)
        pie.slices[i].labelRadius = 1.25
        pie.slices[i].fontSize = 7
    d.add(pie)
    return d

# ── BAR CHART ─────────────────────────────────────────────
def build_bar_chart(categories, values, color=BLUE, width=200, height=120):
    d = Drawing(width, height)
    bc = VerticalBarChart()
    bc.x = 30; bc.y = 20
    bc.width = width - 50; bc.height = height - 40
    bc.data = [values]
    bc.categoryAxis.categoryNames = categories
    bc.categoryAxis.labels.fontSize = 7
    bc.categoryAxis.labels.angle = 0
    bc.valueAxis.labels.fontSize = 7
    bc.bars[0].fillColor = color
    bc.bars[0].strokeColor = None
    d.add(bc)
    return d

# ══════════════════════════════════════════════════════════
# REPORT 1 — Individual Patient Clinical Report
# ══════════════════════════════════════════════════════════
def generate_patient_report(patient: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=15*mm, bottomMargin=20*mm)
    S = get_styles()
    story = []
    W = A4[0] - 40*mm

    # Header
    story.append(build_header_banner(
        "OSA Clinical Patient Report",
        f"Patient ID: {patient.get('id','N/A')}  |  OSA Clinical Decision Support System"
    ))
    story.append(Spacer(1, 8))

    # Risk badge + score row
    risk  = patient.get("risk", "Unknown")
    score = patient.get("pred_score", 0) * 100
    clr   = RISK_COLORS.get(risk, BLUE)

    risk_data = [[
        risk_badge_drawing(risk, score),
        "",
        metric_box("AHI Index",   f"{patient.get('ahi',0):.1f}", RED),
        metric_box("SpO2 (%)",    f"{patient.get('spo2',0)}%",
                   GREEN if patient.get('spo2',100) >= 90 else RED),
        metric_box("BMI",         f"{patient.get('bmi',0):.1f}", AMBER),
        metric_box("ML Score",    f"{score:.0f}%", clr),
    ]]
    risk_tbl = Table(risk_data, colWidths=[35*mm, 5*mm, 30*mm, 30*mm, 30*mm, 30*mm])
    risk_tbl.setStyle(TableStyle([
        ("ALIGN",     (0,0), (-1,-1), "CENTER"),
        ("VALIGN",    (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",(0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1),4),
    ]))
    story.append(risk_tbl)
    story.append(Spacer(1, 8))

    # Clinical recommendation box
    if score >= 75:
        rec_text = "URGENT: High OSA probability detected. Refer immediately for overnight polysomnography (PSG). Evaluate for CPAP/BiPAP therapy. Screen for cardiovascular comorbidities."
        rec_color = RED
    elif score >= 55:
        rec_text = "HIGH RISK: Significant OSA indicators present. Schedule sleep study within 2 weeks. Begin positional therapy and weight management counselling."
        rec_color = colors.HexColor("#DC2626")
    elif score >= 35:
        rec_text = "MODERATE RISK: Schedule sleep study within 4-6 weeks. Advise lifestyle modifications including positional therapy and weight management."
        rec_color = AMBER
    else:
        rec_text = "LOW RISK: Routine annual screening recommended. Advise patient to report changes in daytime somnolence, snoring frequency, or sleep quality."
        rec_color = GREEN

    rec_tbl = Table([[Paragraph(f"<b>Clinical Recommendation:</b> {rec_text}", S["BodyText"])]],
        colWidths=[W])
    rec_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(0,0), colors.HexColor("#FFF7ED") if score >= 35 else colors.HexColor("#F0FDF4")),
        ("LEFTPADDING",  (0,0),(0,0), 8),
        ("RIGHTPADDING", (0,0),(0,0), 8),
        ("TOPPADDING",   (0,0),(0,0), 8),
        ("BOTTOMPADDING",(0,0),(0,0), 8),
        ("LINEBEFOREEACH",(0,0),(0,0), 3, rec_color),
        ("BOX",          (0,0),(0,0), 0.5, GREY_LIGHT),
        ("ROUNDEDCORNERS",(0,0),(0,0), 4),
    ]))
    story.append(rec_tbl)
    story.append(Spacer(1, 10))

    # Patient demographics & vitals tables side by side
    story.append(HRFlowable(width=W, color=GREY_LIGHT, thickness=1))
    story.append(Paragraph("PATIENT DEMOGRAPHICS & CLINICAL PARAMETERS", S["SectionTitle"]))

    def info_table(rows, title):
        data = [[Paragraph(title, ParagraphStyle("th",
            fontSize=8, textColor=WHITE, fontName="Helvetica-Bold",
            alignment=TA_LEFT)), ""]]
        for k, v in rows:
            data.append([
                Paragraph(k, ParagraphStyle("k", fontSize=9, textColor=GREY_MID,
                    fontName="Helvetica", alignment=TA_LEFT)),
                Paragraph(str(v), ParagraphStyle("v", fontSize=9, textColor=NAVY,
                    fontName="Helvetica-Bold", alignment=TA_RIGHT)),
            ])
        tbl = Table(data, colWidths=[45*mm, 35*mm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(1,0), NAVY),
            ("SPAN",          (0,0),(1,0)),
            ("TOPPADDING",    (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 5),
            ("LEFTPADDING",   (0,0),(-1,-1), 8),
            ("RIGHTPADDING",  (0,0),(-1,-1), 8),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),
             [colors.HexColor("#F8FAFC"), colors.white]),
            ("LINEBELOW",     (0,1),(-1,-1), 0.3, GREY_LIGHT),
            ("BOX",           (0,0),(-1,-1), 0.5, GREY_LIGHT),
        ]))
        return tbl

    left_tbl = info_table([
        ("Patient ID",   patient.get("id","N/A")),
        ("Age",          f"{patient.get('age','N/A')} years"),
        ("Gender",       patient.get("gender","N/A")),
        ("BMI",          f"{patient.get('bmi',0):.1f} kg/m²"),
        ("Neck Circumference", f"{patient.get('neck',0)} cm"),
        ("Snoring Index",f"{patient.get('snoring',0)} / 5"),
    ], "Demographics & Anthropometrics")

    right_tbl = info_table([
        ("AHI Index",    f"{patient.get('ahi',0):.1f} events/hr"),
        ("SpO2",         f"{patient.get('spo2',0)}%"),
        ("Blood Pressure",patient.get("bp","N/A")),
        ("Smoker",       "Yes" if patient.get("smoker",0) else "No"),
        ("Diabetes",     "Yes" if patient.get("diabetes",0) else "No"),
        ("Risk Level",   risk),
    ], "Clinical Indicators")

    side_by_side = Table([[left_tbl, Spacer(8,1), right_tbl]],
        colWidths=[80*mm, 8*mm, 80*mm])
    side_by_side.setStyle(TableStyle([
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("TOPPADDING",(0,0),(-1,-1),0),
        ("BOTTOMPADDING",(0,0),(-1,-1),0),
        ("LEFTPADDING",(0,0),(-1,-1),0),
        ("RIGHTPADDING",(0,0),(-1,-1),0),
    ]))
    story.append(side_by_side)
    story.append(Spacer(1, 12))

    # AHI Classification table
    story.append(HRFlowable(width=W, color=GREY_LIGHT, thickness=1))
    story.append(Paragraph("AHI SEVERITY CLASSIFICATION (AASM GUIDELINES)", S["SectionTitle"]))

    ahi_val = patient.get("ahi", 0)
    ahi_data = [
        ["AHI Range", "Classification", "Recommended Action", "Patient Status"],
        ["< 5 /hr",    "Normal",         "No treatment",         ""],
        ["5 – 14 /hr", "Mild OSA",       "Lifestyle changes",    ""],
        ["15 – 29 /hr","Moderate OSA",   "CPAP therapy",         ""],
        [">= 30 /hr",  "Severe OSA",     "Urgent CPAP/BiPAP",    ""],
    ]
    ranges   = [(0,5),(5,15),(15,30),(30,999)]
    row_idx  = next((i+1 for i,(lo,hi) in enumerate(ranges) if lo <= ahi_val < hi), 1)
    labels   = ["Normal","Mild OSA","Moderate OSA","Severe OSA"]
    ahi_data[row_idx][3] = f"CURRENT: {ahi_val:.1f}"

    ahi_tbl = Table(ahi_data, colWidths=[35*mm, 40*mm, 60*mm, 33*mm])
    ahi_style = [
        ("BACKGROUND",    (0,0),(-1,0), NAVY),
        ("TEXTCOLOR",     (0,0),(-1,0), WHITE),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 8),
        ("ALIGN",         (0,0),(-1,-1), "CENTER"),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#F8FAFC"), WHITE]),
        ("LINEBELOW",     (0,0),(-1,-1), 0.3, GREY_LIGHT),
        ("BOX",           (0,0),(-1,-1), 0.5, GREY_LIGHT),
        ("BACKGROUND",    (0,row_idx),(-1,row_idx), colors.HexColor("#FEF9C3")),
        ("TEXTCOLOR",     (3,row_idx),(3,row_idx), RED),
        ("FONTNAME",      (3,row_idx),(3,row_idx), "Helvetica-Bold"),
    ]
    ahi_tbl.setStyle(TableStyle(ahi_style))
    story.append(ahi_tbl)
    story.append(Spacer(1, 10))

    # Footer disclaimer
    story.append(HRFlowable(width=W, color=GREY_LIGHT, thickness=1))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "CLINICAL DISCLAIMER: This report is generated by an AI-assisted decision support tool and does not replace "
        "full polysomnography (PSG) evaluation. All findings must be reviewed and validated by a qualified clinician.",
        S["Disclaimer"]))
    story.append(Paragraph(
        f"OSA-CDSS v2.0  |  Alliance University  |  Dept. of Data Science & Sleep Medicine  |  "
        f"Report generated: {datetime.datetime.now().strftime('%d %b %Y %H:%M')}",
        S["Footer"]))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════
# REPORT 2 — Population / Cohort Analytics Report
# ══════════════════════════════════════════════════════════
def generate_population_report(df: pd.DataFrame, model_metrics: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=15*mm, bottomMargin=20*mm)
    S = get_styles()
    story = []
    W = A4[0] - 40*mm

    # Header
    story.append(build_header_banner(
        "OSA Population Analytics Report",
        f"Cohort: {len(df)} patients  |  Alliance University Hospital Sleep Medicine"
    ))
    story.append(Spacer(1, 10))

    # Model performance metrics row
    story.append(Paragraph("MODEL PERFORMANCE METRICS", S["SectionTitle"]))
    metrics_row = [[
        metric_box("Accuracy",    model_metrics.get("accuracy","N/A"),   BLUE,   35*mm),
        metric_box("Sensitivity", model_metrics.get("sensitivity","N/A"),GREEN,  35*mm),
        metric_box("Specificity", model_metrics.get("specificity","N/A"),AMBER,  35*mm),
        metric_box("AUC-ROC",     model_metrics.get("auc","N/A"),        PURPLE, 35*mm),
        metric_box("Total Patients", str(len(df)),                        NAVY,  28*mm),
    ]]
    m_tbl = Table(metrics_row, colWidths=[35*mm, 35*mm, 35*mm, 35*mm, 28*mm])
    m_tbl.setStyle(TableStyle([
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1),3),
        ("RIGHTPADDING",(0,0),(-1,-1),3),
    ]))
    story.append(m_tbl)
    story.append(Spacer(1, 10))

    # Risk distribution summary
    story.append(HRFlowable(width=W, color=GREY_LIGHT, thickness=1))
    story.append(Paragraph("RISK STRATIFICATION SUMMARY", S["SectionTitle"]))

    risk_counts = df["risk"].value_counts().to_dict() if "risk" in df.columns else {}
    risk_order  = ["Low","Moderate","High","Severe"]

    risk_summary = [["Risk Level","Count","Percentage","Avg AHI","Avg BMI","Avg SpO2"]]
    for r in risk_order:
        sub = df[df["risk"]==r] if "risk" in df.columns else pd.DataFrame()
        count = len(sub)
        pct   = f"{count/len(df)*100:.1f}%" if len(df) > 0 else "0%"
        avg_ahi = f"{sub['ahi'].mean():.1f}"  if not sub.empty and "ahi"  in sub else "N/A"
        avg_bmi = f"{sub['bmi'].mean():.1f}"  if not sub.empty and "bmi"  in sub else "N/A"
        avg_spo = f"{sub['spo2'].mean():.1f}%" if not sub.empty and "spo2" in sub else "N/A"
        risk_summary.append([r, str(count), pct, avg_ahi, avg_bmi, avg_spo])

    rs_tbl = Table(risk_summary, colWidths=[35*mm,25*mm,30*mm,30*mm,30*mm,30*mm])
    rs_style = [
        ("BACKGROUND",    (0,0),(-1,0), NAVY),
        ("TEXTCOLOR",     (0,0),(-1,0), WHITE),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 8),
        ("ALIGN",         (0,0),(-1,-1), "CENTER"),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#F8FAFC"), WHITE]),
        ("LINEBELOW",     (0,0),(-1,-1), 0.3, GREY_LIGHT),
        ("BOX",           (0,0),(-1,-1), 0.5, GREY_LIGHT),
    ]
    for i, r in enumerate(risk_order):
        clr = RISK_COLORS.get(r, BLUE)
        rs_style.append(("TEXTCOLOR", (0, i+1), (0, i+1), clr))
        rs_style.append(("FONTNAME",  (0, i+1), (0, i+1), "Helvetica-Bold"))
    rs_tbl.setStyle(TableStyle(rs_style))
    story.append(rs_tbl)
    story.append(Spacer(1, 10))

    # Charts row: pie + bar
    story.append(HRFlowable(width=W, color=GREY_LIGHT, thickness=1))
    story.append(Paragraph("VISUAL ANALYTICS", S["SectionTitle"]))

    pie = build_pie_chart(
        {r: risk_counts.get(r,0) for r in risk_order if risk_counts.get(r,0) > 0}
    )
    bar_cats = risk_order
    bar_vals = tuple(risk_counts.get(r,0) for r in risk_order)
    bar = build_bar_chart(bar_cats, bar_vals, BLUE, width=195, height=130)

    charts_row = [[pie, bar]]
    c_tbl = Table(charts_row, colWidths=[85*mm, 85*mm])
    c_tbl.setStyle(TableStyle([
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("BOX",(0,0),(-1,-1),0.5,GREY_LIGHT),
        ("TOPPADDING",(0,0),(-1,-1),8),
        ("BOTTOMPADDING",(0,0),(-1,-1),8),
    ]))
    story.append(c_tbl)
    story.append(Spacer(1, 10))

    # Full patient table
    story.append(HRFlowable(width=W, color=GREY_LIGHT, thickness=1))
    story.append(Paragraph("FULL PATIENT REGISTRY", S["SectionTitle"]))

    cols = ["id","age","gender","bmi","ahi","spo2","risk","pred_score"]
    col_labels = ["ID","Age","Gender","BMI","AHI","SpO2","Risk","ML Score"]
    col_widths = [22*mm,15*mm,20*mm,18*mm,18*mm,18*mm,22*mm,22*mm]

    table_data = [col_labels]
    for _, row in df[cols].iterrows():
        table_data.append([
            str(row["id"]),
            str(row["age"]),
            str(row["gender"]),
            f"{row['bmi']:.1f}",
            f"{row['ahi']:.1f}",
            f"{row['spo2']}%",
            str(row["risk"]),
            f"{row['pred_score']*100:.0f}%",
        ])

    pt_tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
    pt_style = [
        ("BACKGROUND",    (0,0),(-1,0), NAVY),
        ("TEXTCOLOR",     (0,0),(-1,0), WHITE),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 7.5),
        ("ALIGN",         (0,0),(-1,-1), "CENTER"),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#F8FAFC"), WHITE]),
        ("LINEBELOW",     (0,0),(-1,-1), 0.3, GREY_LIGHT),
        ("BOX",           (0,0),(-1,-1), 0.5, GREY_LIGHT),
    ]
    # Colour risk column
    for i, (_, row) in enumerate(df[cols].iterrows()):
        clr = RISK_COLORS.get(row["risk"], BLUE)
        pt_style.append(("TEXTCOLOR",  (6, i+1), (6, i+1), clr))
        pt_style.append(("FONTNAME",   (6, i+1), (6, i+1), "Helvetica-Bold"))
    pt_tbl.setStyle(TableStyle(pt_style))
    story.append(pt_tbl)
    story.append(Spacer(1, 10))

    # Footer
    story.append(HRFlowable(width=W, color=GREY_LIGHT, thickness=1))
    story.append(Spacer(1,4))
    story.append(Paragraph(
        "CLINICAL DISCLAIMER: This report is AI-assisted and does not replace clinical diagnosis.",
        S["Disclaimer"]))
    story.append(Paragraph(
        f"OSA-CDSS v2.0  |  Alliance University  |  Population Report  |  "
        f"Generated: {datetime.datetime.now().strftime('%d %b %Y %H:%M')}",
        S["Footer"]))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════
# REPORT 3 — Risk Assessment Result Report
# ══════════════════════════════════════════════════════════
def generate_assessment_report(params: dict, score: float, risk: str) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=15*mm, bottomMargin=20*mm)
    S = get_styles()
    story = []
    W = A4[0] - 40*mm
    clr = RISK_COLORS.get(risk, BLUE)

    story.append(build_header_banner(
        "OSA Risk Assessment Result",
        f"ML-Powered Assessment  |  Ensemble Model (RF + GBM)  |  AUC 1.000"
    ))
    story.append(Spacer(1, 10))

    # Big score display
    score_pct = score * 100
    score_tbl = Table([[
        risk_badge_drawing(risk, score_pct, width=140, height=60),
        Spacer(10,1),
        metric_box("OSA Probability", f"{score_pct:.1f}%", clr, 40*mm, 26*mm),
        metric_box("Classification",  risk,                clr, 40*mm, 26*mm),
    ]], colWidths=[50*mm, 10*mm, 40*mm, 40*mm])
    score_tbl.setStyle(TableStyle([
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ]))
    story.append(score_tbl)
    story.append(Spacer(1, 10))

    # Input parameters table
    story.append(HRFlowable(width=W, color=GREY_LIGHT, thickness=1))
    story.append(Paragraph("INPUT CLINICAL PARAMETERS", S["SectionTitle"]))

    param_data = [
        ["Parameter", "Value", "Clinical Range", "Status"],
        ["Age",          f"{params.get('age','N/A')} yrs",  "18–90 years",  "Normal"],
        ["BMI",          f"{params.get('bmi','N/A'):.1f} kg/m²",
         "18.5–24.9 Normal", "Overweight" if params.get('bmi',0) >= 25 else "Normal"],
        ["Neck Circumference", f"{params.get('neck','N/A')} cm",
         "<40 cm (M), <35 cm (F)",
         "Elevated" if params.get('neck',0) >= 40 else "Normal"],
        ["Snoring Intensity", f"{params.get('snoring','N/A')} / 5",
         "1 = Mild, 5 = Severe",
         "Severe" if params.get('snoring',0) >= 4 else "Moderate" if params.get('snoring',0) == 3 else "Mild"],
        ["Resting SpO2",  f"{params.get('spo2','N/A')}%",   "95–100%",
         "Low" if params.get('spo2',100) < 95 else "Normal"],
        ["Gender",        params.get('gender','N/A'),        "—",            "—"],
        ["Diabetes",      "Yes" if params.get('diabetes') else "No", "—",
         "Risk Factor" if params.get('diabetes') else "None"],
        ["Smoker",        "Yes" if params.get('smoker') else "No", "—",
         "Risk Factor" if params.get('smoker') else "None"],
    ]

    p_tbl = Table(param_data, colWidths=[48*mm, 38*mm, 52*mm, 30*mm])
    p_style = [
        ("BACKGROUND",    (0,0),(-1,0), NAVY),
        ("TEXTCOLOR",     (0,0),(-1,0), WHITE),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 8.5),
        ("ALIGN",         (0,0),(-1,-1), "CENTER"),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#F8FAFC"), WHITE]),
        ("LINEBELOW",     (0,0),(-1,-1), 0.3, GREY_LIGHT),
        ("BOX",           (0,0),(-1,-1), 0.5, GREY_LIGHT),
    ]
    # Colour status column
    for i in range(1, len(param_data)):
        status = param_data[i][3]
        s_clr = RED if status in ("Elevated","Low","Severe","Risk Factor") else \
                AMBER if status in ("Overweight","Moderate") else GREEN
        p_style.append(("TEXTCOLOR",  (3,i),(3,i), s_clr))
        p_style.append(("FONTNAME",   (3,i),(3,i), "Helvetica-Bold"))
    p_tbl.setStyle(TableStyle(p_style))
    story.append(p_tbl)
    story.append(Spacer(1, 10))

    # Clinical recommendation
    story.append(HRFlowable(width=W, color=GREY_LIGHT, thickness=1))
    story.append(Paragraph("CLINICAL RECOMMENDATION", S["SectionTitle"]))

    recs = {
        "Severe": [
            "Immediate referral for overnight polysomnography (PSG)",
            "Urgent CPAP/BiPAP titration study",
            "Cardiovascular risk assessment (ECG, BP monitoring)",
            "Endocrinology referral if diabetic comorbidity present",
            "Follow-up within 2 weeks post-diagnosis",
        ],
        "High": [
            "Referral for overnight PSG within 1–2 weeks",
            "Initiate weight management programme",
            "Blood pressure monitoring (3x weekly)",
            "Consider auto-titrating CPAP (APAP) trial",
            "Review all sedative/hypnotic medications",
        ],
        "Moderate": [
            "Schedule sleep study within 4–6 weeks",
            "Advise positional therapy (lateral sleeping)",
            "Weight reduction target: 5–10% body weight",
            "Avoid alcohol and sedatives within 4 hours of sleep",
            "Annual review if symptoms worsen",
        ],
        "Low": [
            "Annual OSA screening recommended",
            "Maintain healthy BMI (18.5–24.9 kg/m²)",
            "Report changes in snoring or daytime somnolence",
            "Sleep hygiene counselling",
        ],
    }
    rec_list = recs.get(risk, recs["Low"])
    rec_data = [[Paragraph(f"  {i+1}.  {r}",
        ParagraphStyle("rec", fontSize=9, textColor=GREY_DARK,
            fontName="Helvetica", leading=14))]
        for i, r in enumerate(rec_list)]

    rec_tbl = Table(rec_data, colWidths=[W])
    rec_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(0,-1), colors.HexColor("#FFF7ED") if score >= 0.35 else colors.HexColor("#F0FDF4")),
        ("LEFTPADDING",  (0,0),(0,-1), 10),
        ("RIGHTPADDING", (0,0),(0,-1), 10),
        ("TOPPADDING",   (0,0),(0,-1), 4),
        ("BOTTOMPADDING",(0,0),(0,-1), 4),
        ("LINEBEFOREEACH",(0,0),(0,-1), 3, clr),
        ("BOX",          (0,0),(0,-1), 0.5, GREY_LIGHT),
    ]))
    story.append(rec_tbl)
    story.append(Spacer(1, 10))

    # AHI reference
    story.append(HRFlowable(width=W, color=GREY_LIGHT, thickness=1))
    story.append(Paragraph("AHI CLASSIFICATION REFERENCE (AASM 2023)", S["SectionTitle"]))
    ahi_ref = Table([
        ["AHI Range","Classification","Action Required"],
        ["< 5 /hr",   "Normal",       "No intervention needed"],
        ["5–14 /hr",  "Mild OSA",     "Lifestyle modifications"],
        ["15–29 /hr", "Moderate OSA", "CPAP therapy indicated"],
        [">= 30 /hr", "Severe OSA",   "Urgent CPAP/BiPAP"],
    ], colWidths=[40*mm, 50*mm, 78*mm])
    ahi_ref.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), NAVY),
        ("TEXTCOLOR",     (0,0),(-1,0), WHITE),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 8.5),
        ("ALIGN",         (0,0),(-1,-1), "CENTER"),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#F8FAFC"), WHITE]),
        ("LINEBELOW",     (0,0),(-1,-1), 0.3, GREY_LIGHT),
        ("BOX",           (0,0),(-1,-1), 0.5, GREY_LIGHT),
    ]))
    story.append(ahi_ref)
    story.append(Spacer(1, 10))

    # Footer
    story.append(HRFlowable(width=W, color=GREY_LIGHT, thickness=1))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "DISCLAIMER: This AI-generated assessment assists clinical decision-making only. "
        "It does not replace full polysomnography (PSG) or a qualified clinician's judgment.",
        S["Disclaimer"]))
    story.append(Paragraph(
        f"OSA-CDSS v2.0  |  Alliance University  |  Ensemble Model: RF + GBM  |  "
        f"Generated: {datetime.datetime.now().strftime('%d %b %Y %H:%M')}",
        S["Footer"]))

    doc.build(story)
    buf.seek(0)
    return buf.read()
