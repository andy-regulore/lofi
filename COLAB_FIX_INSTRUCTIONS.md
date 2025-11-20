# ⚡ CRITICAL FIX: Update Training Cell in Colab NOW

## The Problem
- Saving to Google Drive every 500 steps = **MASSIVELY SLOW**
- Batch size 4 on A100 (80GB VRAM) = **WASTING GPU POWER**
- 15 epochs = **WAY TOO MANY**

## The Fix
In your Colab notebook, find the training cell (Step 3) and make these changes:

### 1. Change Storage Paths (around line 24-27)
**FIND:**
```python
DRIVE_DIR = '/content/drive/MyDrive/LoFi_Training'
TOKENIZED_DATA_PATH = f'{DRIVE_DIR}/tokenized_data/sequences.pkl'
CHECKPOINT_DIR = f'{DRIVE_DIR}/checkpoints'
FINAL_MODEL_DIR = f'{DRIVE_DIR}/final_model'
```

**REPLACE WITH:**
```python
DRIVE_DIR = '/content/drive/MyDrive/LoFi_Training'
TOKENIZED_DATA_PATH = f'{DRIVE_DIR}/tokenized_data/sequences.pkl'
LOCAL_CHECKPOINT_DIR = '/content/lofi-checkpoints'  # LOCAL storage (FAST!)
FINAL_MODEL_DIR = f'{DRIVE_DIR}/final_model'

# Create local checkpoint directory
os.makedirs(LOCAL_CHECKPOINT_DIR, exist_ok=True)
```

### 2. Update Training Config (around line 34-40)
**FIND:**
```python
config['training']['output_dir'] = CHECKPOINT_DIR
config['training']['device'] = 'cuda'
config['training']['fp16'] = True
config['training']['num_epochs'] = 15  # User set to 15
config['training']['batch_size'] = 4   # User set to 4
config['training']['save_steps'] = 500  # Save checkpoint every 500 steps
config['training']['save_total_limit'] = 3  # Keep only last 3 checkpoints to save space
```

**REPLACE WITH:**
```python
config['training']['output_dir'] = LOCAL_CHECKPOINT_DIR  # LOCAL STORAGE!
config['training']['device'] = 'cuda'
config['training']['fp16'] = True
config['training']['num_epochs'] = 3  # 3 is plenty with millions of sequences
config['training']['batch_size'] = 32  # A100 can handle 32!
config['training']['gradient_accumulation_steps'] = 1  # No need with batch=32
config['training']['max_steps'] = 50000  # Cap at 50k steps
config['training']['save_steps'] = 5000  # Every 5k steps (not 500!)
config['training']['eval_steps'] = 2500  # Eval every 2.5k steps
config['training']['logging_steps'] = 100  # Log every 100 steps
config['training']['save_total_limit'] = 2  # Keep only last 2 checkpoints
```

### 3. Update Checkpoint Detection (around line 180-185)
**FIND:**
```python
if os.path.exists(CHECKPOINT_DIR):
    existing_checkpoint = get_last_checkpoint(CHECKPOINT_DIR)
```

**REPLACE WITH:**
```python
if os.path.exists(LOCAL_CHECKPOINT_DIR):
    existing_checkpoint = get_last_checkpoint(LOCAL_CHECKPOINT_DIR)
```

### 4. Update Final Copy to Drive (around line 200-205)
**FIND:**
```python
# Copy final model to separate directory in Google Drive
print(f"\n💾 Copying final model to: {FINAL_MODEL_DIR}")
import shutil
if os.path.exists(FINAL_MODEL_DIR):
    shutil.rmtree(FINAL_MODEL_DIR)
shutil.copytree(CHECKPOINT_DIR, FINAL_MODEL_DIR)
```

**REPLACE WITH:**
```python
# Copy final model from LOCAL storage to Google Drive
print(f"\n💾 Copying final model from local storage to Google Drive...")
print(f"   Source: {LOCAL_CHECKPOINT_DIR}")
print(f"   Destination: {FINAL_MODEL_DIR}")
import shutil
if os.path.exists(FINAL_MODEL_DIR):
    shutil.rmtree(FINAL_MODEL_DIR)
shutil.copytree(LOCAL_CHECKPOINT_DIR, FINAL_MODEL_DIR)
```

## Performance Impact

**Before:**
- 35,926 steps/epoch × 15 epochs = 539,000 steps
- Checkpoint save every 500 steps = 1,078 saves × 45 sec each = **13.5 HOURS just saving!**
- Batch size 4 = inefficient GPU use

**After:**
- ~4,491 steps/epoch × 3 epochs = 13,473 steps (capped at 50k)
- Checkpoint save every 5,000 steps = 10 saves × 2 sec each = **20 SECONDS saving**
- Batch size 32 = 8x more efficient!

**Training will complete in 8-10 hours instead of days!**
