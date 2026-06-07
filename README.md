# 🌊 Marine Oil Spill Prediction Dashboard

> **AI-Powered Forecasting of Global Oil Spill Trends Using Machine Learning**

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Data](https://img.shields.io/badge/Data%20Period-1970--2023-orange?style=for-the-badge)](Oil%20Spills%20global%20data.csv)

---

## 📋 Overview

This project uses **Machine Learning** to predict the number of large and medium oil spills globally for future years (2025-2034) based on 50+ years of historical data (1970-2023). 

The application provides:
- 📊 **Interactive visual analytics** with Plotly charts
- 🔮 **AI-powered predictions** using Linear Regression
- 📈 **Historical trend analysis** with easy-to-understand visualizations
- 💡 **Clean, modern web interface** built with Streamlit
- 🎯 **Year-by-year forecasting** for planning and awareness

---

## ✨ Features

### 🔮 **Interactive Prediction Engine**
- Select any year (2025-2034) to get predictions
- View predictions for **both large and medium oil spills**
- Predictions table with all years and totals

### 📊 **Advanced Data Visualization**
- **Fully interactive charts** with Plotly (hover, zoom, pan)
- Historical trends from 1970-2023
- Future predictions with visual trends
- Year-by-year comparison charts
- Toggle data series on/off in legend

### 📈 **Comprehensive Analytics**
- Historical averages and statistics
- Latest year data overview
- Trend analysis and insights
- Multi-year comparison tools

### 🎨 **Modern UI/UX**
- Gradient design with professional styling
- Responsive layout (works on all devices)
- Color-coded predictions (Red: Large, Blue: Medium)
- Expandable information sections
- Clear warnings and disclaimers

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** (tested with 3.12)
- **pip** (Python package manager)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Anu929/Oil-Spill-Detection.git
cd Oil-Spill-Detection
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Running the App

**Option 1: Direct Command**
```bash
streamlit run app.py
```

**Option 2: Using Python Module**
```bash
python -m streamlit run app.py
```

**Option 3: Debug & Run (with diagnostics)**
```bash
python debug_and_run.py
```

**Option 4: Windows Batch File**
```bash
run.bat
```

The app will open automatically at: **`http://localhost:8501`**

---

## 📁 Project Structure

```
Oil-Spill-Detection/
├── app.py                          # Main Streamlit application
├── debug_and_run.py               # Debug script with dependency checks
├── requirements.txt               # Python dependencies
├── Oil Spills global data.csv     # Historical data (1970-2023)
├── run.bat                        # Windows launcher script
├── QUICKSTART.md                  # Quick setup guide
└── README.md                      # This file
```

---

## 📊 Dashboard Sections

### 1️⃣ **About Section**
Learn what the dashboard does and its key features.

### 2️⃣ **Prediction Section**
- Slider to select year (2025-2034)
- Real-time predictions for large and medium spills
- Color-coded results for easy understanding

### 3️⃣ **Historical Data Overview**
- Key metrics (latest year, averages, total data)
- Data table with last 10 years of historical data

### 4️⃣ **Interactive Trends Chart**
- Historical data from 1970-2023
- AI predictions for future years
- Fully interactive (zoom, pan, hover for details)
- Toggle data series using legend

### 5️⃣ **Predictions Table**
All predictions for 2025-2034 with totals for planning and comparison.

### 6️⃣ **Comparison Chart**
Select specific years and compare spill incidents side-by-side with bar charts.

### 7️⃣ **Key Insights**
Historical statistics and latest year data at a glance.

---

## 📈 Understanding the Data

### Data Source
- **Global Oil Spill Database** (1970-2023)
- Covers worldwide marine oil spill incidents
- Classified into: Large (>700 tonnes) and Medium (7-700 tonnes)

### Historical Trends
- **Large Spills**: Average ~11 incidents/year
- **Medium Spills**: Average ~37 incidents/year
- Clear downward trend over recent decades (better regulations & technology)

### How Predictions Work
The app uses **Linear Regression** (machine learning) to:
1. Analyze 50+ years of historical data
2. Identify trends and patterns
3. Project future incidents based on those patterns
4. Generate predictions for 2025-2034

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit 1.28+ |
| **Data Processing** | Pandas 2.1+ |
| **Machine Learning** | Scikit-learn 1.3+ |
| **Visualization** | Plotly 5.17+ |
| **Computation** | NumPy 1.26+ |
| **Language** | Python 3.8+ |

---

## 📋 Requirements

```
streamlit>=1.28.0
pandas>=2.1.0
scikit-learn>=1.3.0
plotly>=5.17.0
numpy>=1.26.0
```

See `requirements.txt` for exact versions.

---

## 🐛 Troubleshooting

### Issue: Port 8501 already in use
```bash
streamlit run app.py --server.port 8502
```

### Issue: Module not found errors
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Issue: CSV file not found
- Ensure you're in the project directory
- Check that `Oil Spills global data.csv` exists in the same folder as `app.py`

### Issue: Python not recognized
- Make sure Python is in your PATH
- Try `python --version` to verify installation

---

## ⚠️ Important Disclaimers

- **These predictions are AI-generated estimates** based on historical trends
- **Actual incidents depend on many factors:**
  - Environmental practices and regulations
  - Technological improvements
  - Economic factors
  - Enforcement and monitoring
  - Climate conditions

- **Use these forecasts as a reference tool,** not absolute predictions
- For critical planning, consult industry experts and regulatory bodies

---

## 📊 Sample Output

### Prediction Example
```
Year: 2030
Large Spill Incidents: 8
Medium Spill Incidents: 28
Total Incidents: 36
```

### Historical Context
```
Year: 2023 (Latest)
Large Spills: 3
Medium Spills: 16
```

---

## 🤝 Contributing

We welcome contributions! Areas for enhancement:
- [ ] Add more sophisticated ML models (Polynomial, ARIMA, Neural Networks)
- [ ] Include regional breakdowns
- [ ] Add data export functionality (CSV, PDF)
- [ ] Implement confidence intervals
- [ ] Add historical events context
- [ ] Create mobile-friendly version

To contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📞 Support

- **Issues?** Open a GitHub issue
- **Questions?** Check QUICKSTART.md for setup help
- **Feature requests?** Create a discussion or issue

---

## 🎯 Future Enhancements

- [ ] Real-time data updates from APIs
- [ ] Regional prediction models
- [ ] Risk assessment scoring
- [ ] Alert system for high-risk predictions
- [ ] Export reports (PDF, Excel)
- [ ] Mobile app version
- [ ] Multi-language support
- [ ] Collaborative notes & analysis

---

## 📚 References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Global Oil Spill Database](https://ourworldindata.org/)

---

<div align="center">

### 🌊 Made with ❤️ for Ocean Protection

**Help predict, monitor, and reduce oil spills worldwide!**

[⭐ Star this project](https://github.com/Anu929/Oil-Spill-Detection) if you found it helpful

</div>

---

**Last Updated:** June 2024  
**Version:** 2.0 (Interactive Predictions Edition)