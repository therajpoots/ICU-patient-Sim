"""
PDF Report Generator Module
Compiles a clinical case study PDF for doctors containing:
1. Patient metadata.
2. Grouped table of all logged anomalies.
3. Decrypted waveform charts (ECG, PPG, RSP) plotted dynamically.
4. DeepSeek-powered medical analysis and diagnostic insights.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend to prevent PyQt6 thread/backend conflicts
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
from openai import OpenAI

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Rule-based clinical summaries for offline fallback
OFFLINE_SUMMARIES = {
    "sinus_tachycardia": (
        "Clinical Assessment: Sinus Tachycardia detected. Patient shows heart rate significantly above baseline. "
        "Potential causes: pain, fever, anxiety, or hypovolemia. Monitor fluid status and rule out infection."
    ),
    "sinus_bradycardia": (
        "Clinical Assessment: Sinus Bradycardia detected. Heart rate is abnormally low. "
        "Observe for signs of poor perfusion (lethargy, hypotensive trend). Re-evaluate beta-blocker dosage."
    ),
    "pvc": (
        "Clinical Assessment: Premature Ventricular Contractions (PVC) detected. "
        "Occasional ectopy is common in ICU, but high burden could trigger re-entry arrhythmias. Check serum potassium and magnesium levels."
    ),
    "atrial_fibrillation": (
        "Clinical Assessment: Atrial Fibrillation (AFib) with rapid ventricular response. "
        "Irregular heart rhythm detected, causing reduced diastolic filling. High risk of thromboembolism. "
        "Review anticoagulation plan and consider rate/rhythm control therapy."
    ),
    "spo2_desaturation": (
        "Clinical Assessment: SpO₂ Desaturation (Hypoxia). Oxygen saturation dropped below 90%. "
        "Potential causes: mucous plug, atelectasis, respiratory depression, or ventilator misalignment. "
        "Provide supplemental oxygen and check airway patency immediately."
    ),
    "hypertensive_spike": (
        "Clinical Assessment: Hypertensive Spike. Systolic BP exceeded critical threshold. "
        "Clinical risks: acute kidney injury, encephalopathy, or intracranial hemorrhage. "
        "Check pain management and consider short-acting intravenous vasodilators."
    ),
    "respiratory_distress": (
        "Clinical Assessment: Respiratory Distress. Tachypnea and dynamic hyperinflation markers. "
        "Assess accessory muscle use, obtain arterial blood gases, and rule out pulmonary edema or bronchospasm."
    ),
    "ventricular_fibrillation": (
        "Clinical Assessment: VENTRICULAR FIBRILLATION (Cardiac Arrest). Chaotic ventricular telemetry with absolute "
        "hemodynamic collapse (blood pressure collapsed to shock floor of 25/15 mmHg) and loss of peripheral pulse (PPG flatline). "
        "This is a medical emergency. Initiate immediate CPR and prepare for defibrillation."
    ),
}

def query_deepseek_insight(state: str, vitals: Dict[str, Any], api_key: str) -> str:
    """Queries DeepSeek API to get a clinical insight for the physician."""
    if not api_key or api_key == "sk-3ae47177f18e4ecf808440d6168c0d6f":
        return OFFLINE_SUMMARIES.get(state, "Clinical Assessment: Anomaly detected. Monitor patient vitals closely.")
        
    shap_contribs = vitals.get("shap_contributors", [])
    shap_str = ""
    if shap_contribs:
        shap_str = ", driven by: " + ", ".join([f"{item['feature']} ({item['influence']:+.2f})" for item in shap_contribs])

    prompt = (
        f"You are a cardiologist and ICU physician. Generate a concise, highly professional clinical assessment "
        f"for a patient chart. The patient experienced a '{state}' anomaly{shap_str}. "
        f"Vitals during the episode: HR={vitals.get('heart_rate')} bpm, BP={vitals.get('systolic_bp')}/{vitals.get('diastolic_bp')} mmHg, "
        f"SpO2={vitals.get('spo2')}%, RR={vitals.get('respiratory_rate')} breaths/min, "
        f"Core Temp={vitals.get('core_temperature')} °C, Skin Temp={vitals.get('skin_temperature')} °C. "
        f"Focus on clinical significance, potential causes, acute risks, and immediate medical suggestions. "
        f"Limit your response to 4 sentences."
    )
    
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a clinical ICU physician writing patient records."},
            {"role": "user", "content": prompt}
        ],
        timeout=3.0
    )
    return response.choices[0].message.content

def generate_pdf_report(
    patient_info: Dict[str, Any],
    anomaly_logs: List[Dict[str, Any]],
    output_path: str,
    api_key: str = "sk-3ae47177f18e4ecf808440d6168c0d6f"
):
    """Compiles the clinical PDF report incorporating patient logs, plots, and DeepSeek insights."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )
    
    use_offline = False
    
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#111827'),
        alignment=1, # Center
        spaceAfter=15
    )
    
    section_style = ParagraphStyle(
        'SecTitle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#06b6d4'),
        spaceBefore=12,
        spaceAfter=8
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#374151')
    )
    
    clinical_notes_style = ParagraphStyle(
        'ClinicalNotes',
        parent=body_style,
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#7c3aed') # Purple tone for AI insights
    )
    
    meta_style = ParagraphStyle(
        'MetaText',
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#1f2937')
    )

    story = []

    # 1. Header
    story.append(Paragraph("ICU Bedside Telemetry Anomaly Report", title_style))
    story.append(Spacer(1, 10))

    # 2. Patient Demographics Block
    p = patient_info
    meta_data = [
        [Paragraph(f"<b>Patient Name:</b> {p.get('patient_name', 'John Doe')}", meta_style),
         Paragraph(f"<b>Patient ID:</b> {p.get('patient_id', 'N/A')}", meta_style)],
        [Paragraph(f"<b>Ward/Bed:</b> {p.get('ward', 'N/A')}", meta_style),
         Paragraph(f"<b>Age / Sex:</b> {p.get('patient_age', 'N/A')} y.o / Male", meta_style)],
        [Paragraph(f"<b>Report Time:</b> {time.strftime('%Y-%m-%d %H:%M:%S')}", meta_style),
         Paragraph(f"<b>Total Anomalies:</b> {len(anomaly_logs)}", meta_style)]
    ]
    t_meta = Table(meta_data, colWidths=[270, 270])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f3f4f6')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#e5e7eb')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 20))

    # 3. Overall Anomalies List
    story.append(Paragraph("Anomaly Episode Summary", section_style))
    
    summary_headers = ["Time", "Duration", "Arrhythmia / Deterioration", "Vitals during Peak"]
    summary_data = [summary_headers]
    
    for idx, log in enumerate(reversed(anomaly_logs)):
        start_t = time.strftime('%H:%M:%S', time.localtime(log["start_time"]))
        dur = f"{log['end_time'] - log['start_time']:.0f}s"
        v = log.get("vitals")
        if not isinstance(v, dict):
            v = {}
        hr_val = v.get("heart_rate")
        sbp_val = v.get("systolic_bp")
        dbp_val = v.get("diastolic_bp")
        spo2_val = v.get("spo2")
        
        hr_str = f"{hr_val:.0f}" if hr_val is not None else "N/A"
        sbp_str = f"{sbp_val:.0f}" if sbp_val is not None else "N/A"
        dbp_str = f"{dbp_val:.0f}" if dbp_val is not None else "N/A"
        spo2_str = f"{spo2_val:.0f}" if spo2_val is not None else "N/A"
        
        v_str = f"HR:{hr_str} | BP:{sbp_str}/{dbp_str} | SpO₂:{spo2_str}%"
        
        # Human readable label
        state_label = log["state"].replace("_", " ").title()
        if log["state"] == "vfib":
            state_label = "Ventricular Fibrillation (VFIB)"
        elif log["state"] == "pvc":
            state_label = "PVC Ectopy"
            
        summary_data.append([start_t, dur, state_label, v_str])
        
    t_summary = Table(summary_data, colWidths=[80, 70, 190, 200])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1f2937')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#d1d5db')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('FONTSIZE', (0,1), (-1,-1), 8.5),
        ('BACKGROUND', (0,1), (-1,-1), colors.white),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 15))
    
    # 4. Detailed Anomaly telemetry sheets
    story.append(PageBreak())
    
    temp_plots = []
    
    for idx, log in enumerate(reversed(anomaly_logs)):
        state_label = log["state"].replace("_", " ").title()
        start_t_full = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(log["start_time"]))
        
        # Create KeepTogether block so the charts, vitals, and notes stay on the same page
        section_elements = []
        
        section_elements.append(Paragraph(f"Telemetry Detail {idx+1}: {state_label} (onset: {start_t_full})", section_style))
        section_elements.append(Spacer(1, 5))
        
        # Plot Waveforms
        w = log.get("waveforms")
        if not isinstance(w, dict):
            w = {}
        ecg_w = w.get("ecg") or []
        ppg_w = w.get("ppg") or []
        rsp_w = w.get("rsp") or []
        
        fs = 250.0
        max_len = max(len(ecg_w), len(ppg_w), len(rsp_w))
        t = np.arange(max_len) / fs
        
        # Ensure all segments are matched to the maximum length (pad with 0.0)
        ecg_data = list(ecg_w) + [0.0] * (max_len - len(ecg_w))
        ppg_data = list(ppg_w) + [0.0] * (max_len - len(ppg_w))
        rsp_data = list(rsp_w) + [0.0] * (max_len - len(rsp_w))
        
        fig, axes = plt.subplots(3, 1, figsize=(7, 4), sharex=True)
        # ECG
        axes[0].plot(t, ecg_data, color='#10b981', lw=1)
        axes[0].set_ylabel('ECG', color='#10b981', fontsize=8, fontweight='bold')
        axes[0].tick_params(axis='y', labelsize=6)
        axes[0].grid(True, linestyle='--', alpha=0.3)
        # PPG
        axes[1].plot(t, ppg_data, color='#06b6d4', lw=1)
        axes[1].set_ylabel('PPG', color='#06b6d4', fontsize=8, fontweight='bold')
        axes[1].tick_params(axis='y', labelsize=6)
        axes[1].grid(True, linestyle='--', alpha=0.3)
        # RSP
        axes[2].plot(t, rsp_data, color='#f59e0b', lw=1)
        axes[2].set_ylabel('RSP', color='#f59e0b', fontsize=8, fontweight='bold')
        axes[2].set_xlabel('Time (s)', fontsize=8, fontweight='bold')
        axes[2].tick_params(axis='both', labelsize=6)
        axes[2].grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plot_filename = f"temp_plot_report_{log['id']}_{int(time.time())}.png"
        plt.savefig(plot_filename, dpi=200)
        plt.close()
        temp_plots.append(plot_filename)
        
        # Embed Plot Image
        section_elements.append(Image(plot_filename, width=480, height=240))
        section_elements.append(Spacer(1, 10))
        
        # Query DeepSeek for Medical Assessment
        insight = None
        if not use_offline and api_key and api_key != "sk-3ae47177f18e4ecf808440d6168c0d6f":
            try:
                insight = query_deepseek_insight(log["state"], log["vitals"], api_key)
            except Exception as e:
                print(f"DeepSeek connection error: {e}. Falling back to offline assessment for remainder of report.")
                use_offline = True
                
        if not insight:
            insight = OFFLINE_SUMMARIES.get(log["state"], "Clinical Assessment: Anomaly detected. Monitor patient vitals closely.")
        
        v = log.get("vitals")
        if not isinstance(v, dict):
            v = {}
        v_headers = ["BP", "SpO2", "HR", "Resp. Rate", "Core / Skin Temp"]
        
        sbp_val = v.get('systolic_bp')
        dbp_val = v.get('diastolic_bp')
        spo2_val = v.get('spo2')
        hr_val = v.get('heart_rate')
        rr_val = v.get('respiratory_rate')
        temp_c_val = v.get('core_temperature')
        temp_s_val = v.get('skin_temperature')
        
        bp_str = f"{sbp_val:.1f}/{dbp_val:.1f} mmHg" if (sbp_val is not None and dbp_val is not None) else "N/A"
        spo2_str = f"{spo2_val:.1f}%" if spo2_val is not None else "N/A"
        hr_str = f"{hr_val:.1f} bpm" if hr_val is not None else "N/A"
        rr_str = f"{rr_val:.1f} breaths/min" if rr_val is not None else "N/A"
        temp_str = f"{temp_c_val:.2f}°C / {temp_s_val:.2f}°C" if (temp_c_val is not None and temp_s_val is not None) else "N/A"
        
        v_row = [bp_str, spo2_str, hr_str, rr_str, temp_str]
        
        t_v_details = Table([v_headers, v_row], colWidths=[110, 75, 75, 110, 110])
        t_v_details.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f3f4f6')),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e5e7eb')),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8.5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 4),
        ]))
        
        section_elements.append(t_v_details)
        section_elements.append(Spacer(1, 8))
        
        # Add SHAP explanation
        shap_contribs = v.get("shap_contributors", [])
        if shap_contribs:
            shap_text_items = []
            for item in shap_contribs:
                f_name = item["feature"].replace("_", " ").title()
                inf = item["influence"]
                shap_text_items.append(f"<b>{f_name}</b> ({inf:+.2f})")
            shap_text = ", ".join(shap_text_items)
            section_elements.append(Paragraph(f"<b>ML SHAP Contributing Drivers:</b> {shap_text}", body_style))
            section_elements.append(Spacer(1, 8))
        
        # Add AI Insight
        section_elements.append(Paragraph("<b>Clinical Notes & Assessment:</b>", body_style))
        section_elements.append(Paragraph(insight, clinical_notes_style))
        section_elements.append(Spacer(1, 20))
        
        # Keep each detail section together on a single page if possible
        story.append(KeepTogether(section_elements))
        
        # Page break between detail sheets
        if idx < len(anomaly_logs) - 1:
            story.append(PageBreak())

    # Build PDF
    doc.build(story)
    
    # Cleanup temp plot images
    for plot in temp_plots:
        try:
            if os.path.exists(plot):
                os.remove(plot)
        except Exception as e:
            print(f"Error cleaning up plot file {plot}: {e}")
            
    print(f"PDF clinical report successfully generated at: {output_path}")
