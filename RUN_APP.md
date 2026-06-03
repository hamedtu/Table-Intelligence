# How to Run the Streamlit App

## Quick Start

### Step 1: Open Terminal
Navigate to the project folder in your terminal:
```powershell
cd C:\my_project  (example)
```

### Step 2: Install Dependencies 
Run this command to install all required packages:
```powershell
pip install -r requirements.txt
```

### Step 3: Start the App
Run this command to launch the Streamlit app:
```powershell
streamlit run app.py
```

Your browser will automatically open to **http://localhost:8501**

---

## Stopping the App

Press `Ctrl + C` in the terminal where Streamlit is running.

---

## Troubleshooting

### "pip command not found"
Make sure Python is installed. Run:
```powershell
python --version
```

### "streamlit: command not found"
Dependencies not installed. Run:
```powershell
pip install -r requirements.txt
```

### "ModuleNotFoundError"
Try reinstalling:
```powershell
pip install -r requirements.txt --force-reinstall
```

### App doesn't open browser
Manually visit: **http://localhost:8501**

### Demo images not showing
Make sure the `tables/` folder exists in the project directory.

---


