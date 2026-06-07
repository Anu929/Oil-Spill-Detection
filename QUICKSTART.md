# 🌊 Quick Start Guide

## Option 1: Using Batch File (Easiest) ✅
1. Double-click `run.bat`
2. The app will install dependencies and start automatically
3. Open your browser to `http://localhost:8501`

---

## Option 2: Manual Terminal Commands
```bash
# Navigate to the project directory
cd Oil-Spill-Detection

# Install dependencies
pip install -r requirements.txt

# Run debug checks and start app
python debug_and_run.py
```

---

## Option 3: Direct Streamlit Command
```bash
# Install dependencies first
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 🐛 Troubleshooting

### Issue: "pip command not found"
- Make sure Python is installed and added to PATH
- Try: `python -m pip install -r requirements.txt`

### Issue: "streamlit command not found"
- Run: `python -m streamlit run app.py`

### Issue: CSV file not found
- Make sure you're in the correct directory
- Check that `Oil Spills global data.csv` exists in the same folder

### Issue: Port 8501 already in use
```bash
streamlit run app.py --server.port 8502
```

### Issue: ModuleNotFoundError
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

---

## 📊 App Features
✅ Interactive Plotly charts (zoom, pan, hover)  
✅ Year selector (2025-2034)  
✅ Predictions for large AND medium spills  
✅ Side-by-side prediction display  
✅ Historical data comparison  
✅ Key insights & analytics  
✅ Modern gradient design  
✅ Easy-to-understand explanations  

---

## 🌐 Browser Access
Once running, open: **http://localhost:8501**

Press `Ctrl+C` in terminal to stop the app.

---

## 📋 New Features (v2.0)
- ✨ Fully interactive Plotly charts
- 🔮 Dual predictions (large + medium spills)
- 📊 Side-by-side prediction boxes
- 📈 Multi-year comparison chart
- 🎯 Comprehensive predictions table
- 💡 Historical insights section
- 🎨 Professional gradient design

---

## 📞 Support
If you encounter issues:
1. Check TROUBLESHOOTING section above
2. Verify Python version: `python --version`
3. Check dependencies: `pip list`
4. Review README.md for more details
