"""Transparent, non-clinical wellness calculations used by the POC."""


def calculate_bmi(height_cm, weight_kg):
    """Return BMI, or None when the entered measurements are unusable."""
    if not height_cm or not weight_kg or height_cm <= 0 or weight_kg <= 0:
        return None
    return round(weight_kg / ((height_cm / 100) ** 2), 1)


def calculate_bmi_category(bmi):
    if bmi is None:
        return "Not available"
    if bmi < 18.5:
        return "Below selected reference range"
    if bmi < 25:
        return "Selected reference range"
    if bmi < 30:
        return "Above selected reference range"
    return "Well above selected reference range"


def calculate_sleep_score(hours, quality, awakenings, fatigue):
    """POC 0-100 score: duration 35%, perceived quality 30%, awakenings 20%, fatigue 15%."""
    duration = max(0, 100 - min(abs((hours or 0) - 8) * 25, 100))
    quality_score = ((quality or 1) - 1) / 4 * 100
    awakening_score = max(0, 100 - min((awakenings or 0) * 25, 100))
    fatigue_score = ((5 - (fatigue or 5)) / 4) * 100
    return round(0.35 * duration + 0.30 * quality_score + 0.20 * awakening_score + 0.15 * fatigue_score)


def calculate_wellbeing_score(stress, energy):
    """POC 0-100 score: lower self-reported stress and higher energy contribute equally."""
    stress_score = (5 - (stress or 5)) / 4 * 100
    energy_score = ((energy or 1) - 1) / 4 * 100
    return round((stress_score + energy_score) / 2)


def calculate_cardio_indicators(age, bmi, smoking, activity, systolic_bp=None, diastolic_bp=None, resting_hr=None):
    """Return cautious, self-report-based indicators; this is not a risk or disease calculator."""
    factors = []
    points = 0
    if bmi is not None and (bmi < 18.5 or bmi >= 25):
        points += 1; factors.append("BMI is outside the selected reference range")
    if smoking == "Current":
        points += 2; factors.append("Current smoking reported")
    elif smoking == "Former":
        points += 1; factors.append("Former smoking reported")
    if activity in ("Low", "None"):
        points += 1; factors.append("Lower activity reported")
    if resting_hr is not None and (resting_hr < 50 or resting_hr > 100):
        points += 1; factors.append("Camera-derived resting HR is outside a broad reference range")
    if systolic_bp and systolic_bp >= 130 or diastolic_bp and diastolic_bp >= 80:
        points += 1; factors.append("Entered blood pressure is above the selected reference range")
    if not factors:
        factors.append("No selected indicators were reported")
    label = "Favourable" if points == 0 else "Attention Suggested" if points <= 2 else "Multiple Risk Indicators"
    return {"label": label, "factors": factors, "indicator_count": points}


def calculate_vitality_score(activity, sleep_score, bmi, smoking, resting_hr, wellbeing_score):
    """POC composite: activity, sleep, BMI, smoking, HR and wellbeing are equally weighted when present."""
    parts = []
    activity_map = {"High": 100, "Moderate": 75, "Low": 40, "None": 15}
    if activity in activity_map: parts.append(("Activity", activity_map[activity]))
    if sleep_score is not None: parts.append(("Sleep", sleep_score))
    if wellbeing_score is not None: parts.append(("Wellbeing", wellbeing_score))
    if bmi is not None: parts.append(("BMI", 100 if 18.5 <= bmi < 25 else 65 if 25 <= bmi < 30 else 35))
    if smoking: parts.append(("Smoking", {"Never": 100, "Former": 75, "Current": 25}.get(smoking, 50)))
    if resting_hr is not None: parts.append(("Resting HR", 100 if 50 <= resting_hr <= 85 else 65 if 45 <= resting_hr <= 100 else 35))
    score = round(sum(value for _, value in parts) / len(parts)) if parts else None
    return {"score": score, "components": parts}
