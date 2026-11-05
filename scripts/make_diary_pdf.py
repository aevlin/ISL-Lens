import sys
from pathlib import Path
import pymupdf
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

PAGE_WIDTH, PAGE_HEIGHT = letter  # 612 x 792 pt
MARGIN = 36  # 0.5 inch

def generate_diary_pdf(output_path, font_size=9.0, leading=11.5, pad_v1=4.8, pad_v2=3.4, p1_count=5, sp_after_title=12, sp_eval_timeline=14, spacer_sig=55):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=26,
        bottomMargin=22
    )

    title_style = ParagraphStyle(
        'DocTitle',
        fontName='Times-Bold',
        fontSize=11.5,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.black
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        fontName='Times-Bold',
        fontSize=10.5,
        leading=13,
        alignment=TA_CENTER,
        textColor=colors.black
    )
    # The user requested: "They're heading on Project Diary ISL-LENS up to November 2026. Should be on the same line."
    # 9.5pt in Times-Bold is 517pt wide, perfectly fitting the 540pt width on a SINGLE line.
    meta_banner_style = ParagraphStyle(
        'MetaBanner',
        fontName='Times-Bold',
        fontSize=9.5,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.black
    )
    
    info_style = ParagraphStyle(
        'InfoStyle',
        fontName='Times-Roman',
        fontSize=9.0,
        leading=12.5,
        alignment=TA_LEFT,
        textColor=colors.black
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        fontName='Times-Bold',
        fontSize=9.5,
        leading=12.5,
        alignment=TA_LEFT,
        textColor=colors.black
    )
    
    th_style = ParagraphStyle(
        'TableHeader',
        fontName='Times-Bold',
        fontSize=font_size,
        leading=leading,
        alignment=TA_CENTER,
        textColor=colors.black
    )
    
    td_center = ParagraphStyle(
        'TableCellCenter',
        fontName='Times-Roman',
        fontSize=font_size,
        leading=leading,
        alignment=TA_CENTER,
        textColor=colors.black
    )
    td_left = ParagraphStyle(
        'TableCellLeft',
        fontName='Times-Roman',
        fontSize=font_size,
        leading=leading,
        alignment=TA_LEFT,
        textColor=colors.black
    )

    story = []

    # Header
    story.append(Paragraph("DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING", title_style))
    story.append(Spacer(1, 2))
    story.append(Paragraph("23CSD415 PROJECT PHASE I — Project Diary", subtitle_style))
    story.append(Spacer(1, 2.5))
    # Full line bold up to November 2026 on the EXACT SAME SINGLE LINE
    story.append(Paragraph("<b>ISL-LENS: Real-Time Indian Sign Language Recognition And Interactive AI Console | 2023–2027 Batch | July 2026 – Nov 2026</b>", meta_banner_style))
    story.append(Spacer(1, 5))

    # Student metadata table in straight vertical order with bold person name and guide name
    meta_table_data = [
        [Paragraph("<b>Name:</b> <b>AEVLIN PRINCE</b>", info_style)],
        [Paragraph("<b>Roll No.:</b> 13", info_style)],
        [Paragraph("<b>Reg. No.:</b> AJC23CS014", info_style)],
        [Paragraph("<b>Guide:</b> <b>Dr.Indu John</b>, Asst. Professor, Dept. of CSE", info_style)],
        [Paragraph("<b>Project Commencement:</b> 14-07-2026", info_style)],
        [Paragraph("<b>Project Title:</b> <b>ISL-LENS:</b> Real-Time Indian Sign Language Recognition And Interactive AI Console", info_style)]
    ]
    meta_table = Table(meta_table_data, colWidths=[540])
    meta_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0.6),
        ('TOPPADDING', (0, 0), (-1, -1), 0.6),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(meta_table)

    # USER REQUEST: "And after the project title, there should be a little gap between the official evaluation schedule and the project title."
    story.append(Spacer(1, sp_after_title))

    # Table 1: Official Evaluation Schedule
    story.append(Paragraph("<b>Official Evaluation Schedule (for reference — from the 23CSD415 Project Phase I circular)</b>", section_heading))
    story.append(Spacer(1, 3))

    t1_headers = [
        Paragraph("<b>Sr.</b>", th_style),
        Paragraph("<b>Date / Period</b>", th_style),
        Paragraph("<b>Official Milestone (23CSD415 Project Phase I)</b>", th_style),
        Paragraph("<b>Marks</b>", th_style)
    ]
    t1_rows = [
        t1_headers,
        [
            Paragraph("1", td_center),
            Paragraph("6.7.2026 – 7.7.2026", td_center),
            Paragraph("Discussion with guide and finalizing the topic. Daily/weekly activity diary maintained.", td_left),
            Paragraph("10", td_center)
        ],
        [
            Paragraph("2", td_center),
            Paragraph("8.7.2026 – 14.7.2026", td_center),
            Paragraph("Project Proposal presentation. Abstract + base paper (SCI/Scopus, 2025/26) submitted to shared folder.", td_left),
            Paragraph("—", td_center)
        ],
        [
            Paragraph("3", td_center),
            Paragraph("3.8.2026 – 14.8.2026", td_center),
            Paragraph("Interim Review 1 — literature survey, requirement analysis, feasibility study, system design, 20% implementation.", td_left),
            Paragraph("20", td_center)
        ],
        [
            Paragraph("4", td_center),
            Paragraph("14.9.2026 – 25.9.2026", td_center),
            Paragraph("Interim Review 2 — 50% implementation, module demonstration.", td_left),
            Paragraph("20", td_center)
        ],
        [
            Paragraph("5", td_center),
            Paragraph("26.10.2026 – 28.10.2026", td_center),
            Paragraph("Final Review — 80% implementation, 10-min presentation + 10-min demo. Report submitted 26.10.2026 (10) + Coordinator (10).", td_left),
            Paragraph("30+10+10", td_center)
        ]
    ]

    t1 = Table(t1_rows, colWidths=[30, 98, 354, 58])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#d9d9d9')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), pad_v1),
        ('BOTTOMPADDING', (0, 0), (-1, -1), pad_v1),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t1)

    # Gap between evaluation schedule and development timeline
    story.append(Spacer(1, sp_eval_timeline))

    # Section 2: Development Timeline
    story.append(Paragraph("<b>Development Timeline</b>", section_heading))
    story.append(Spacer(1, 4))

    timeline_header = [
        Paragraph("<b>Week / Month</b>", th_style),
        Paragraph("<b>Development Work to be Done & Progress Status</b>", th_style),
        Paragraph("<b>Remarks</b>", th_style),
        Paragraph("<b>Signature of Project guide with Date</b>", th_style)
    ]

    all_weeks = [
        (
            "<b>Week 1</b><br/>02–06 Jul 2026",
            "Project topic discussed with guide Dr. Indu John and finalized on 06 July 2026. Defined project objectives, technical scope, baseline feasibility, and target Deaf and Hard-of-Hearing (D/HOH) end-user requirements.",
            False
        ),
        (
            "<b>Week 2</b><br/>13–17 Jul 2026",
            "Project proposal presentation prepared and delivered on 14 July 2026. Presented hardware-free webcam architecture, isolated/continuous ISL targets, literature benchmark comparison, and project milestone plan.",
            False
        ),
        (
            "<b>Week 3</b><br/>20–24 Jul 2026",
            "Conducted comprehensive literature survey on vision-based sign recognition, base paper analysis (SCI/Scopus 2025/26), MediaPipe Holistic spatial coordinate normalization, and open-source ISL datasets.",
            False
        ),
        (
            "<b>Week 4</b><br/>27–31 Jul 2026",
            "Drafted System Requirements Specification (SRS) and feasibility study. Finalized client-side browser execution pipeline, 30 FPS camera processing constraints, and zero-dependency vanilla JS architecture.",
            False
        ),
        (
            "<b>Week 5</b><br/>10–14 Aug 2026<br/><b>Interim Review 1<br/>(20 Marks)</b>",
            "Demonstrated system architecture, literature survey, SRS, and initial landmark extraction prototype during Interim Review 1 on 13 August 2026. Approximately 20% implementation verified and approved.",
            True
        ),
        (
            "<b>Week 6</b><br/>17–21 Aug 2026",
            "Organized and preprocessed ISL video datasets (INCLUDE, CISLR, ISL500) across 40 vocabulary classes. Built automated keypoint extraction pipeline using MediaPipe Holistic (pose, face, hands).",
            False
        ),
        (
            "<b>Week 7</b><br/>24–28 Aug 2026",
            "Implemented 2D landmark spatial coordinate normalization, upper-body/shoulder anchoring, and feature cache generation. Trained baseline multinomial logistic regression classifier on landmark vectors.",
            False
        ),
        (
            "<b>Week 8</b><br/>31 Aug–04 Sep 2026",
            "Tuned multi-class sign recognition model across 40 classes. Implemented calibrated confidence thresholding (0.70) and dynamic open-set UNKNOWN rejection filter to eliminate background gesture noise.",
            False
        ),
        (
            "<b>Week 9</b><br/>07–11 Sep 2026",
            "Developed web-native frontend interface using HTML5, CSS3, and JavaScript. Integrated live browser webcam feed, real-time landmark canvas overlay, and client-side inference pipeline.",
            False
        ),
        (
            "<b>Week 10</b><br/>14–18 Sep 2026",
            "Constructed continuous sentence builder module, Subject-Object-Verb (SOV) to Subject-Verb-Object (SVO) grammar reordering rules, and Web Speech API speech synthesis integration.",
            False
        ),
        (
            "<b>Week 11</b><br/>21–25 Sep 2026<br/><b>Interim Review 2<br/>(20 Marks)</b>",
            "Demonstrated live webcam sign recognition, real-time sentence construction, and text-to-speech synthesis during Interim Review 2 on 24 September 2026. 50% project implementation verified.",
            True
        ),
        (
            "<b>Week 12</b><br/>28 Sep–02 Oct 2026",
            "Developed Interactive AI Retraining Studio. Enabled users to capture custom webcam gesture exemplars, compute 828-dimensional feature vectors, and update classification weights via validation gate.",
            False
        ),
        (
            "<b>Week 13</b><br/>05–09 Oct 2026",
            "Built Sign Practice Studio featuring authentic dataset reference video player, real-time circular HUD accuracy gauge, and auditory feedback cues. Conducted review on 08 October 2026.",
            False
        ),
        (
            "<b>Week 14</b><br/>12–16 Oct 2026",
            "Conducted quantitative system profiling: achieved 18.4 ms recognition latency (54.3 FPS throughput), 96.88% cross-signer test accuracy, and validated lighting and background invariance.",
            False
        ),
        (
            "<b>Week 15</b><br/>19–23 Oct 2026",
            "Prepared comprehensive project documentation, draft B.Tech report, IJERA journal research paper manuscript, and presentation slides. Rehearsed final evaluation demonstration on 22 October 2026.",
            False
        ),
        (
            "<b>Week 16</b><br/>26–30 Oct 2026<br/><b>Final Review<br/>(30+10+10 Marks)</b>",
            "Delivered final 10-minute presentation and live 10-minute system demonstration before the evaluation committee on 28 October 2026. Successfully defended 80% implementation and submitted report.",
            True
        ),
        (
            "<b>Week 17</b><br/>02–06 Nov 2026",
            "Incorporated evaluation committee feedback, finalized project report and camera-ready journal paper, and formulated Phase II (Semester 8) deep learning and sentence expansion roadmap.",
            False
        )
    ]

    weeks_p1 = all_weeks[:p1_count]
    weeks_p2 = all_weeks[p1_count:]

    # Table on Page 1 (Weeks 1 to 5)
    p1_rows = [timeline_header]
    p1_styles = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#d9d9d9')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), pad_v1),
        ('BOTTOMPADDING', (0, 0), (-1, -1), pad_v1),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]

    for idx, (w_lbl, w_desc, is_review) in enumerate(weeks_p1):
        row_idx = idx + 1
        p1_rows.append([
            Paragraph(w_lbl, td_center),
            Paragraph(w_desc, td_left),
            Paragraph("", td_center),  # blank remarks for guide
            Paragraph("", td_center)   # blank signature with date for guide
        ])
        if is_review:
            p1_styles.append(('BACKGROUND', (0, row_idx), (-1, row_idx), colors.HexColor('#cfcfcf')))

    t2_p1 = Table(p1_rows, colWidths=[88, 250, 106, 96])
    t2_p1.setStyle(TableStyle(p1_styles))
    story.append(t2_p1)

    # Page Break to Page 2
    story.append(PageBreak())

    # Table on Page 2 (Weeks 6 to 17)
    p2_rows = [timeline_header]
    p2_styles = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#d9d9d9')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), pad_v2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), pad_v2),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]

    for idx, (w_lbl, w_desc, is_review) in enumerate(weeks_p2):
        row_idx = idx + 1
        p2_rows.append([
            Paragraph(w_lbl, td_center),
            Paragraph(w_desc, td_left),
            Paragraph("", td_center),  # blank remarks for guide
            Paragraph("", td_center)   # blank signature with date for guide
        ])
        if is_review:
            p2_styles.append(('BACKGROUND', (0, row_idx), (-1, row_idx), colors.HexColor('#cfcfcf')))

    t2_p2 = Table(p2_rows, colWidths=[88, 250, 106, 96])
    t2_p2.setStyle(TableStyle(p2_styles))
    story.append(t2_p2)

    story.append(Spacer(1, spacer_sig))

    # Signatures block pushed down
    sig_data = [
        [
            Paragraph("_______________________________<br/><b>Signature of Student</b>", ParagraphStyle('SigLeft', fontName='Times-Bold', fontSize=9.5, leading=15, alignment=TA_CENTER)),
            "",
            Paragraph("_______________________________<br/><b>Signature of Project guide</b>", ParagraphStyle('SigRight', fontName='Times-Bold', fontSize=9.5, leading=15, alignment=TA_CENTER))
        ]
    ]
    sig_table = Table(sig_data, colWidths=[240, 60, 240])
    sig_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(sig_table)

    doc.build(story)
    doc_check = pymupdf.open(str(output_path))
    print(f"Generated {output_path} with {len(doc_check)} pages.")
    return len(doc_check)

if __name__ == '__main__':
    generate_diary_pdf("Deliverables/Project_Diary.pdf")
