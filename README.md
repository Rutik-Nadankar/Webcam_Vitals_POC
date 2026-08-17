# Contactless Health Assessment POC

Local Streamlit demonstration that combines camera-derived rPPG/shoulder-motion signals with self-reported wellness inputs. It is not a medical device.

## Run

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

Upload a 45–90 second, well-lit MP4 with a mostly steady face and visible upper chest/shoulders. The local `haarcascade_frontalface_default.xml` is required and deliberately used by the analysis engine.
