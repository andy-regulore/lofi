# 🎨 Customization Guide - Make It Your Own!

Quick guide to customizing your LoFi music generation.

---

## 🎵 Quick Start

**Edit the top of `generate_enhanced.py`:**

```python
# ============================================================================
# CONFIGURATION - EDIT THESE TO CHANGE YOUR MUSIC!
# ============================================================================

MOOD = 'chill'           # Change this!
THEME = 'urban_chill'    # Change this!
LOFI_PRESET = 'medium'   # Change this!
DURATION = 180           # Change this!
KEY = 'Am'               # Change this!
```

Then run: `python generate_enhanced.py`

---

## 🎭 Available Moods

| Mood | Sound | Best For |
|------|-------|----------|
| `'chill'` | Relaxed, easy-going | Study, work, relaxation |
| `'melancholic'` | Nostalgic, reflective | Late night, journaling |
| `'upbeat'` | Energetic, positive | Morning, productivity |
| `'relaxed'` | Calm, peaceful | Meditation, sleep prep |
| `'dreamy'` | Ethereal, floating | Creative work, daydreaming |

**Example:**
```python
MOOD = 'melancholic'  # Nostalgic vibes
```

---

## 🌆 Available Themes (Ambient Sounds)

| Theme | What You Hear | Vibe |
|-------|---------------|------|
| `'rain'` | Rainfall, occasional thunder | Cozy, indoors |
| `'cafe'` | Coffee shop, chatter, espresso machine | Social, warm |
| `'urban_chill'` | City ambience, distant traffic | Metropolitan, modern |
| `'nature'` | Forest, birds, wind | Peaceful, outdoors |
| `'plain'` | No ambient sounds | Pure music |

**Example:**
```python
THEME = 'rain'  # Rainy day vibes
```

---

## 🎛️ LoFi Effect Intensity

| Preset | Effect | Sound |
|--------|--------|-------|
| `'light'` | Subtle | Clean with slight vintage feel |
| `'medium'` | Balanced | Classic lofi sound |
| `'heavy'` | Maximum | Very degraded, cassette tape vibe |

**Example:**
```python
LOFI_PRESET = 'heavy'  # Maximum vintage effect
```

---

## 🎹 Musical Keys

| Key | Mood | Common Use |
|-----|------|------------|
| `'C'` | Bright, simple | Uplifting |
| `'Am'` | Melancholic | Sad, reflective |
| `'F'` | Warm | Comfortable |
| `'G'` | Happy | Positive |
| `'Dm'` | Dark | Serious |
| `'Em'` | Mysterious | Atmospheric |

**Example:**
```python
KEY = 'Dm'  # Darker, more serious
```

---

## ⏱️ Duration

**In seconds:**
```python
DURATION = 60    # 1 minute (short clip)
DURATION = 180   # 3 minutes (standard)
DURATION = 300   # 5 minutes (longer)
DURATION = 600   # 10 minutes (extended)
```

---

## 🎨 Popular Combinations

### 1. **Classic Study Session**
```python
MOOD = 'chill'
THEME = 'plain'
LOFI_PRESET = 'medium'
KEY = 'C'
```

### 2. **Rainy Day Chill**
```python
MOOD = 'melancholic'
THEME = 'rain'
LOFI_PRESET = 'heavy'
KEY = 'Am'
```

### 3. **Coffee Shop Morning**
```python
MOOD = 'upbeat'
THEME = 'cafe'
LOFI_PRESET = 'light'
KEY = 'F'
```

### 4. **Urban Night Drive**
```python
MOOD = 'dreamy'
THEME = 'urban_chill'
LOFI_PRESET = 'medium'
KEY = 'Em'
```

### 5. **Nature Meditation**
```python
MOOD = 'relaxed'
THEME = 'nature'
LOFI_PRESET = 'light'
KEY = 'G'
```

---

## 🔧 Advanced Customization

### Change Chord Progressions

Edit `generate_enhanced.py` around line 60:

```python
chord_progressions = {
    'chill': ([261.63, 196.00, 220.00, 174.61], "C-G-Am-F"),
    'your_custom_mood': ([YOUR_FREQUENCIES], "Chord-Names"),
}
```

### Adjust Ambient Mix Level

Around line 100:

```python
if THEME == 'rain':
    ambient = ambient_gen.generate_rain(DURATION, intensity='medium')
    mix_level = 0.15  # Change this! (0.0 = none, 0.5 = half volume)
```

### Add More Harmonics

Around line 80, add more sine waves:

```python
# Add seventh
audio[start_idx:end_idx] += 0.05 * np.sin(2 * np.pi * freq * 1.75 * segment)
```

---

## 💡 Tips for Best Results

1. **Start with presets** - Use the popular combinations above
2. **Match mood + theme** - Rainy themes work well with melancholic moods
3. **Experiment!** - Try random combinations, you might discover something cool
4. **Generate multiple versions** - Same settings can sound slightly different each time
5. **Save your favorites** - Note down combinations that work well

---

## 🚀 Quick Commands

```powershell
# Generate with current settings
python generate_enhanced.py

# Generate 10 different combinations quickly
# (Edit settings, run, edit settings, run, repeat)

# Check what you created
dir output\audio\

# Play all tracks
start output\audio\
```

---

## 🎵 Example Workflow

1. **Choose your vibe**:
   - Working? → `mood='chill'`, `theme='cafe'`
   - Relaxing? → `mood='relaxed'`, `theme='rain'`
   - Late night? → `mood='melancholic'`, `theme='urban_chill'`

2. **Edit `generate_enhanced.py`** (top 5 lines)

3. **Run it**: `python generate_enhanced.py`

4. **Listen**: `start output\audio\track_*.wav`

5. **Tweak and repeat** until you love it!

---

## ❓ FAQ

**Q: Can I change multiple things at once?**
A: Yes! Change any combination of MOOD, THEME, LOFI_PRESET, KEY, DURATION.

**Q: How do I go back to original settings?**
A: Default settings are at the top - just reset them:
```python
MOOD = 'chill'
THEME = 'urban_chill'
LOFI_PRESET = 'medium'
DURATION = 180
KEY = 'Am'
```

**Q: Can I make it sound more/less lofi?**
A: Yes! Use `LOFI_PRESET = 'light'` for cleaner or `'heavy'` for more degraded sound.

**Q: How do I remove ambient sounds?**
A: Use `THEME = 'plain'`

**Q: Can I make longer tracks?**
A: Yes! Set `DURATION = 600` for 10 minutes, or any number in seconds.

---

**Have fun creating your perfect lofi vibe!** 🎵✨
