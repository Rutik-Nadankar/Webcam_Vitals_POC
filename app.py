from pathlib import Path
import tempfile
import shutil
import streamlit as st
from health_scores import (calculate_bmi, calculate_bmi_category, calculate_sleep_score,
    calculate_wellbeing_score, calculate_vitality_score, calculate_cardio_indicators)
from vitals_engine import analyze_video, create_optimized_demo_copy

BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "videos"

st.set_page_config(page_title="Contactless Health Assessment POC", page_icon="❤", layout="wide")
st.markdown("""<style>.tile{padding:1rem;border:1px solid #e5e7eb;border-radius:12px;background:#fff;min-height:120px}.value{font-size:1.65rem;font-weight:700;color:#0f766e}.muted{color:#64748b;font-size:.9rem}</style>""", unsafe_allow_html=True)
st.title("Contactless Health Assessment POC")
st.caption("Camera-based physiological signals + self-reported wellness indicators")

def card(title, value, note=""):
    st.markdown(f'<div class="tile"><b>{title}</b><div class="value">{value}</div><div class="muted">{note}</div></div>', unsafe_allow_html=True)

def run_camera_analysis(path, force=False):
    progress = st.progress(0, text="Preparing video")
    def update(stage, percent):
        progress.progress(percent, text=f"{stage}: {percent}%")
    if force:
        st.session_state.pop("camera", None)
    st.session_state.camera = analyze_video(path, force_reanalyze=force, progress_callback=update)
    progress.empty()

with st.sidebar:
    st.header("1 — User Profile")
    age=st.number_input("Age", 18, 120, 30); sex=st.selectbox("Sex (optional)", ["Prefer not to say", "Female", "Male", "Another identity"])
    height=st.number_input("Height (cm)", 80.0, 250.0, 170.0); weight=st.number_input("Weight (kg)", 20.0, 300.0, 70.0)
    activity=st.selectbox("Activity level", ["Moderate", "High", "Low", "None"]); smoking=st.selectbox("Smoking status", ["Never", "Former", "Current"])
    cigarettes=st.number_input("Cigarettes/day (if current)", 0, 100, 0)
    st.caption("Optional entered blood pressure")
    systolic=st.number_input("Systolic BP", 0, 260, 0); diastolic=st.number_input("Diastolic BP", 0, 180, 0)
    st.header("2 — Wellness Questionnaire")
    sleep_hours=st.slider("Average sleep (hours)", 0.0, 14.0, 7.5, .5); sleep_quality=st.slider("Perceived sleep quality", 1, 5, 3)
    awakenings=st.slider("Nighttime awakenings", 0, 8, 1); fatigue=st.slider("Daytime tiredness", 1, 5, 3)
    stress=st.slider("Stress level", 1, 5, 3); energy=st.slider("Energy level", 1, 5, 3)

sample_videos = sorted(path for path in VIDEO_DIR.glob("*") if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"})
if sample_videos:
    selected_sample = st.selectbox("Use a local sample video (no upload limit)", sample_videos, format_func=lambda path: f"{path.name} ({path.stat().st_size / 1024**3:.2f} GB)")
    st.caption("This reads the video directly from the local videos folder, so Streamlit's 200 MB upload limit does not apply.")
    left, right = st.columns(2)
    with left:
        if st.button("Analyse local sample video", type="primary"):
            run_camera_analysis(selected_sample)
    with right:
        if st.button("Re-analyse video"):
            run_camera_analysis(selected_sample, force=True)
    if shutil.which("ffmpeg") and st.button("Create optimized demo copy"):
        with st.spinner("Creating optional 720p/30 FPS demo copy..."):
            demo_copy = create_optimized_demo_copy(selected_sample)
        st.success(f"Created: {demo_copy.name}")
else:
    st.info("Add a supported video to the local videos folder to analyse it without uploading.")

st.header("3 — Video")
upload=st.file_uploader("Upload an MP4 video (45–90 seconds preferred)", type=["mp4", "mov", "avi", "mkv"])
if upload and st.button("Analyse video", type="primary"):
    suffix=Path(upload.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(upload.getbuffer()); path=temp.name
    try:
        with st.spinner("Analysing video..."):
            run_camera_analysis(path)
    finally:
        Path(path).unlink(missing_ok=True)

camera=st.session_state.get("camera")
bmi=calculate_bmi(height,weight); sleep_score=calculate_sleep_score(sleep_hours,sleep_quality,awakenings,fatigue); wellbeing=calculate_wellbeing_score(stress,energy)
hr=camera.get("heart_rate") if camera else None
cardio=calculate_cardio_indicators(age,bmi,smoking,activity,systolic or None,diastolic or None,hr)
vitality=calculate_vitality_score(activity,sleep_score,bmi,smoking,hr,wellbeing)

if camera:
    st.divider(); st.subheader(f"Measurement Signal Quality: {camera['signal_quality_label']} — {camera['signal_quality_score']}/100")
    st.caption(f"Face detected in {camera['face_detection_percent']}% of analysed frames · {camera['measurement_duration']} seconds")
    st.header("❤ Vitals")
    a,b,c=st.columns(3)
    with a: card("Heart Rate", f"{hr:.0f} BPM" if hr else "—", f"Signal: {camera['heart_rate_quality']}" if hr else "Insufficient signal")
    with b:
        rr=camera['respiratory_rate']; card("Respiratory Rate", f"{rr:.0f} breaths/min" if rr else "—", f"Signal: {camera['respiratory_quality']}" if rr else "Insufficient signal")
    with c: card("Camera signal", f"{camera['signal_quality_score']}/100", camera['signal_quality_label'])
else:
    st.info("Upload a video to add camera-derived measurements. The questionnaire dashboard below remains available.")

st.divider(); st.header("Dashboard")
cols=st.columns(3)
with cols[0]: card("🫀 Cardiovascular", cardio['label'], " • ".join(cardio['factors']))
with cols[1]: card("🧬 Digital Biomarkers", f"HR: {hr:.0f} BPM" if hr else "HR: —", f"RR: {camera['respiratory_rate']:.0f}" if camera and camera['respiratory_rate'] else "RR: Insufficient signal")
with cols[2]: card("🧠 Wellbeing", f"{wellbeing}/100", f"Stress {stress}/5 · Energy {energy}/5")
with cols[0]: card("⚖️ Physical Health", f"BMI {bmi}" if bmi else "BMI —", f"{calculate_bmi_category(bmi)} · {activity} activity")
with cols[1]: card("🎂 Age / Wellness", f"{age} years", "Chronological age only; no lifespan or biological-age claim.")
with cols[2]: card("😴 Sleep", f"{sleep_score}/100", f"POC wellness score · {sleep_hours} h · quality {sleep_quality}/5")
with cols[0]:
    if camera and camera['rmssd'] is not None: card("💓 HRV — Experimental", f"RMSSD {camera['rmssd']:.0f} ms", f"SDNN {camera['sdnn']:.0f} ms · {camera['accepted_beats']} accepted beats · {camera['hrv_quality']}")
    else: card("💓 HRV — Experimental", "—", "Insufficient signal")
with cols[1]: card("🚬 Smoking", smoking, f"{cigarettes} cigarettes/day" if smoking=="Current" else "User-entered only")
with cols[2]: card("⚡ Vitality", f"{vitality['score']}/100" if vitality['score'] is not None else "—", "POC composite: " + ", ".join(name for name,_ in vitality['components']))
with cols[0]: card("👤 Profile", f"{age} years · {sex}", f"{height:.0f} cm · {weight:.1f} kg")
if camera and camera['warnings']:
    st.warning(" ".join(camera['warnings']))
if camera and camera.get("timing"):
    source_info = camera["timing"]
    if source_info.get("source_width", 0) >= 3000 or source_info.get("source_height", 0) >= 2000 or source_info.get("source_fps", 0) > 30:
        st.info("For faster analysis, a 720p/1080p 30 FPS recording is recommended. Higher-resolution videos provide little benefit for this POC and may decode slowly.")
if camera and camera.get("timing"):
    with st.expander("Technical details"):
        st.json({"performance": camera["timing"], "heart_rate_diagnostics": camera.get("hr_diagnostics", {})})
with st.expander("POC score formulas and safety notes"):
    st.write("Sleep weighs duration (35%), perceived quality (30%), awakenings (20%), and fatigue (15%). Wellbeing equally weighs self-reported stress and energy. Vitality averages available activity, sleep, BMI, smoking, resting HR, and wellbeing components. These are transparent POC wellness scores, not clinically validated measures.")
st.divider(); st.warning("This prototype is for research and demonstration purposes only. It is not a medical device and does not provide medical diagnosis or treatment recommendations.")
