# 🎨 Web UI Quick Start Guide

Beautiful web interface for LoFi Music Empire - no command line needed!

---

## 🚀 Starting the UI

### Step 1: Install Flask (if not already)
```powershell
pip install flask
```

### Step 2: Start the Server
```powershell
python web_ui.py
```

### Step 3: Open Your Browser
```
http://localhost:5000
```

That's it! The UI will open automatically.

---

## 🎵 Using the Web UI

### **Left Panel - Generate Tracks**

1. **Choose Your Settings:**
   - **Mood**: chill, melancholic, upbeat, relaxed, dreamy
   - **Theme**: rain, cafe, urban_chill, nature, or plain
   - **LoFi Effect**: light, medium, or heavy
   - **Key**: C, Am, F, G, Dm, Em
   - **Duration**: 30-600 seconds

2. **Click "Generate Track"**
   - Watch the progress bar
   - Track appears in your library when done

3. **Generation takes ~30-60 seconds**

### **Right Panel - Your Library**

**Statistics at the top:**
- Total Tracks
- Total Size
- Total Duration

**Each track shows:**
- Title (auto-generated)
- Mood, duration, file size
- Three action buttons

**Action Buttons:**
- **ℹ️ Info** - View full metadata (title, description, tags, BPM, key)
- **⬇️ Download** - Download the WAV file to your computer
- **🗑️ Delete** - Remove the track

---

## 🎨 Features

### ✅ Real-Time Progress
- See generation progress live
- Know exactly what step it's on
- No waiting in the dark!

### ✅ Track Management
- View all your tracks at once
- Sort by newest first
- Quick download/delete

### ✅ Full Metadata
- Click "Info" on any track
- See title, description, tags
- View BPM, key, mood

### ✅ Auto-Refresh
- Library refreshes every 10 seconds
- Always shows latest tracks
- No need to reload page

---

## 💡 Quick Tips

### **Fastest Way to Create Multiple Tracks:**
1. Generate first track
2. While it's generating, plan next settings
3. As soon as it's done, change settings and generate again
4. Build your library in minutes!

### **Popular Presets:**

**Study Session:**
- Mood: chill
- Theme: plain
- LoFi: medium
- Key: C

**Rainy Day:**
- Mood: melancholic
- Theme: rain
- LoFi: heavy
- Key: Am

**Coffee Shop:**
- Mood: upbeat
- Theme: cafe
- LoFi: light
- Key: F

**Urban Night:**
- Mood: dreamy
- Theme: urban_chill
- LoFi: medium
- Key: Em

---

## 🔧 Troubleshooting

**Server won't start?**
```powershell
# Make sure Flask is installed
pip install flask

# Check if port 5000 is free
# Try a different port:
# Edit web_ui.py, change last line to:
# app.run(debug=True, host='0.0.0.0', port=5001)
```

**Can't access from another device?**
```powershell
# The server is already set to 0.0.0.0 (all interfaces)
# Find your computer's IP: ipconfig
# On another device: http://YOUR_IP:5000
```

**Generation fails?**
- Check the terminal for error messages
- Make sure output/audio and output/metadata folders exist
- Ensure you have write permissions

**Tracks not appearing?**
- Check output/audio folder
- Refresh your browser
- Restart the server

---

## 📂 File Locations

All generated files are saved in:
```
output/
├── audio/          # WAV audio files
└── metadata/       # JSON metadata files
```

You can access these directly from File Explorer too!

---

## 🎵 Workflow Examples

### **Create a Study Playlist (10 tracks in 10 minutes):**
1. Start server: `python web_ui.py`
2. Open browser: `http://localhost:5000`
3. Set: Mood=chill, Theme=plain, Duration=180
4. Generate first track
5. While generating, prepare next settings
6. Repeat 10 times with slight variations
7. Download all tracks using the download button
8. You now have a 30-minute study playlist!

### **Experiment with Themes:**
1. Use same mood/key but different themes
2. Generate 5 tracks:
   - Theme: plain
   - Theme: rain
   - Theme: cafe
   - Theme: urban_chill
   - Theme: nature
3. Compare and pick your favorite!

---

## 🌐 Accessing from Other Devices

The UI is accessible from any device on your network!

**On your phone/tablet:**
1. Find your computer's IP address:
   ```powershell
   ipconfig
   # Look for IPv4 Address (e.g., 192.168.1.100)
   ```

2. On your phone, open browser:
   ```
   http://192.168.1.100:5000
   ```

3. You can now generate tracks from your phone!

---

## 🎨 UI Features

### Beautiful Design
- Modern gradient interface
- Smooth animations
- Responsive (works on mobile too!)
- Dark mode friendly

### Easy to Use
- No technical knowledge needed
- Everything has helpful tooltips
- Clear labels and descriptions
- Visual progress tracking

### Fast & Efficient
- Background processing
- Real-time updates
- No page reloads needed
- Instant downloads

---

## 🔥 Power User Tips

**Batch Generate:**
Open multiple browser tabs and generate multiple tracks simultaneously!

**Keyboard Shortcuts:**
- Tab through form fields
- Enter to submit
- Esc to close modals

**Download All:**
Open each track's download link in new tabs to download multiple at once.

---

## ❓ FAQ

**Q: Can I use this while generating?**
A: Yes! The UI is fully interactive. Browse your library while generating.

**Q: How many tracks can I store?**
A: Unlimited! Only limited by your hard drive space.

**Q: Can I edit track metadata?**
A: Not in UI yet, but metadata files are JSON - edit with any text editor!

**Q: Does it work offline?**
A: Yes! Everything runs locally. No internet needed.

**Q: Can multiple people use it?**
A: Yes! Share your IP address on your network.

---

## 🎉 That's It!

You now have a complete web UI for creating LoFi music!

**Start generating:**
```powershell
python web_ui.py
```

Then open: **http://localhost:5000**

**Have fun creating! 🎵**
