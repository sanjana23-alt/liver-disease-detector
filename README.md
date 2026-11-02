# Liver disease detector

A website that estimates how closely blood-test values match patients with liver disease.
Trained on the Indian Liver Patient Dataset (UCI). For education only, not medical advice.

## Files
- `indian_liver_patient.csv` the dataset
- `train_model.py` reads the dataset, trains the model, writes `model.js`
- `index.html`, `style.css`, `script.js` the website (uses `model.js`)

## Run
1. `python -m pip install -r requirements.txt`
2. `python train_model.py`   (creates `model.js`)
3. Open `index.html` in a browser.
