#!/usr/bin/env python
import sys
import os

print("=" * 60)
print("🔍 DEBUGGING OIL SPILL DETECTION APP")
print("=" * 60)

# Check Python version
print(f"\n✓ Python Version: {sys.version}")

# Check current directory
current_dir = os.getcwd()
print(f"✓ Current Directory: {current_dir}")

# Check if CSV file exists
csv_file = "Oil Spills global data.csv"
if os.path.exists(csv_file):
    print(f"✓ CSV File Found: {csv_file}")
    # Check file size
    file_size = os.path.getsize(csv_file)
    print(f"  └─ File Size: {file_size} bytes")
else:
    print(f"✗ CSV File NOT Found: {csv_file}")
    sys.exit(1)

# Check required packages
print("\n📦 Checking Dependencies...")
required_packages = ['streamlit', 'pandas', 'sklearn', 'plotly', 'numpy']
missing_packages = []

for package in required_packages:
    try:
        if package == 'sklearn':
            __import__('sklearn')
        else:
            __import__(package)
        print(f"  ✓ {package}")
    except ImportError:
        print(f"  ✗ {package} - MISSING")
        missing_packages.append(package)

if missing_packages:
    print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
    print(f"Install with: pip install {' '.join(missing_packages)}")
    sys.exit(1)

# Try to load and process data
print("\n📊 Testing Data Loading...")
try:
    import pandas as pd
    df = pd.read_csv(csv_file)
    print(f"  ✓ Data loaded successfully")
    print(f"  ✓ Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  ✓ Columns: {list(df.columns)}")
    
    # Clean data as per app.py
    df = df.drop(columns=["Entity", "Code"])
    df.columns = ["Year", "Large_Spills", "Medium_Spills"]
    df["Large_Spills"] = pd.to_numeric(df["Large_Spills"], errors='coerce')
    df["Medium_Spills"] = pd.to_numeric(df["Medium_Spills"], errors='coerce')
    print(f"  ✓ Data cleaned and formatted")
    print(f"  ✓ Year range: {df['Year'].min()} - {df['Year'].max()}")
except Exception as e:
    print(f"  ✗ Error loading data: {e}")
    sys.exit(1)

# Try ML models
print("\n🤖 Testing ML Models...")
try:
    from sklearn.linear_model import LinearRegression
    
    model_large = LinearRegression()
    model_large.fit(df[["Year"]], df["Large_Spills"])
    
    model_medium = LinearRegression()
    model_medium.fit(df[["Year"]], df["Medium_Spills"])
    
    # Test predictions
    test_pred_large = model_large.predict([[2030]])[0]
    test_pred_medium = model_medium.predict([[2030]])[0]
    print(f"  ✓ Models trained successfully")
    print(f"  ✓ Test prediction for 2030:")
    print(f"    - Large Spills: {test_pred_large:.2f} incidents")
    print(f"    - Medium Spills: {test_pred_medium:.2f} incidents")
except Exception as e:
    print(f"  ✗ Error with ML models: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL CHECKS PASSED! Ready to run Streamlit app...")
print("=" * 60)
print("\n🚀 Starting Streamlit app...\n")

# Run streamlit
import subprocess
subprocess.run(["streamlit", "run", "app.py"])
