# Kepler Exoplanet Classifier

A **1D CNN** that classifies Kepler light curves as **CONFIRMED** exoplanets or **FALSE POSITIVE**.

> **No internet needed!**  
> `process_data.py` **uses mock (fake but realistic) data by default** — perfect for demos, classrooms, or offline use.

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/Debug-AstroByte/Exoplanet-Classifier.git
cd Exoplanet-Classifier

# 2. Set up environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate data (uses mock data — no download!)
python process_data.py
# → creates processed_data_output.pkl in <10 seconds

# 5. Run the web demo
streamlit run app.py