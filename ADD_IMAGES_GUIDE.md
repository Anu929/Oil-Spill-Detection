# 📸 How to Add Screenshots to README

## Steps to Add Images

### 1. **Take Screenshots of Your App**
Run `streamlit run app.py` and take screenshots of:
- **Dashboard overview** (main page)
- **Prediction section** (with numbers)
- **Interactive chart** (with zoom/hover visible)
- **Data table** (predictions)
- **Comparison chart**

### 2. **Upload Images to GitHub**

**Option A: Direct Upload via GitHub Web**
```
1. Go to your repository: https://github.com/Anu929/Oil-Spill-Detection
2. Click "Add file" → "Upload files"
3. Drag & drop your screenshots
4. Commit the files to main branch
5. Create a new folder (e.g., `/screenshots` or `/docs`)
```

**Option B: Upload via Terminal**
```bash
# Create screenshots folder
mkdir screenshots

# Copy your images to this folder
# Then commit and push
git add screenshots/
git commit -m "docs: Add dashboard screenshots"
git push
```

### 3. **Add Images to README**

Replace the placeholder sections in `README.md` with actual images:

```markdown
## 📸 Dashboard Preview

### Main Dashboard
![Dashboard Overview](screenshots/dashboard-overview.png)

### Prediction Section
![Prediction Example](screenshots/prediction-section.png)

### Interactive Chart
![Interactive Chart](screenshots/interactive-chart.png)

### Data Table
![Predictions Table](screenshots/predictions-table.png)
```

---

## 📸 Screenshot Recommendations

### What to Capture:

| Section | Size | Format |
|---------|------|--------|
| Full Dashboard | 1280x720 | PNG/JPG |
| Predictions | 1280x500 | PNG/JPG |
| Charts | 1280x600 | PNG/JPG |
| Data Table | 1280x400 | PNG/JPG |

### Tips:
- Use PNG for lossless quality
- Crop to show key features
- Keep file size < 500KB per image
- Use consistent sizing for visual appeal
- Name files descriptively (e.g., `dashboard-overview.png`)

---

## Example Image Markdown

Add this to your README.md after each section:

```markdown
### 🎯 Prediction Engine
*Select any year and get instant predictions for both large and medium spills*

![Prediction Engine](screenshots/prediction-example.png)

### 📊 Interactive Charts
*Fully interactive charts with zoom, pan, and hover functionality*

![Interactive Chart](screenshots/chart-interactive.png)

### 📈 Historical Analysis
*Compare different years and view trends over 50+ years*

![Historical Analysis](screenshots/historical-data.png)
```

---

## ✅ Checklist

- [ ] Take screenshots of all main sections
- [ ] Save as PNG/JPG files
- [ ] Create `/screenshots` folder on GitHub
- [ ] Upload images to the folder
- [ ] Add image links to README.md
- [ ] Test links are working
- [ ] Verify images display correctly

---

## Need Help?

If images aren't showing:
1. Check the path is correct: `screenshots/filename.png`
2. Make sure files are committed and pushed to GitHub
3. Try refreshing the README page
4. Use relative paths, not absolute URLs

---

**Happy documenting!** 🌊📸
