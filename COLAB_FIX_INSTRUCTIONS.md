# LoFi Training - OneDrive Checkpoint Configuration

## Current Configuration

The Colab notebook now uses **OneDrive** for checkpoint storage with optimized settings:

### Storage
- **OneDrive** mounted via rclone at `/content/onedrive`
- Checkpoints saved to `/content/onedrive/LoFi_Training/checkpoints`
- Tokenized data cached at `/content/onedrive/LoFi_Training/tokenized_data`

### Training Settings
```python
config['training']['save_steps'] = 10000    # Save every 10,000 steps
config['training']['save_total_limit'] = 3  # Keep last 3 checkpoints
config['training']['num_epochs'] = 15
config['training']['batch_size'] = 4
config['training']['fp16'] = True
```

## OneDrive Setup

To connect your OneDrive in Colab:

1. **Install rclone locally** and run `rclone config`
2. **Create a new remote** called 'onedrive' for Microsoft OneDrive
3. **Copy your config** from `~/.config/rclone/rclone.conf`
4. **Paste it** in the Colab notebook setup cell

See https://rclone.org/onedrive/ for detailed setup instructions.

## Performance Tips

For faster training on A100 GPU:
- Increase `batch_size` to 32 (A100 can handle it)
- Set `max_steps` to limit training duration
- Reduce `num_epochs` to 3-5
- Increase `save_steps` to reduce checkpoint overhead

## Resuming Training

If Colab disconnects:
1. Re-run the OneDrive mount cells
2. Re-run the training cell
3. Training auto-resumes from the last checkpoint
