# Example Scripts

This directory contains example scripts demonstrating various usage patterns.

## Files

### `basic_workflow.py`
Complete end-to-end workflow for a single track.

```bash
python examples/basic_workflow.py
```

Demonstrates:
- Music generation
- Copyright checking
- Video creation
- Metadata generation
- Thumbnail creation

### `batch_generation.py`
Generate multiple tracks at once.

```bash
python examples/batch_generation.py
```

Demonstrates:
- Batch generation (5 tracks)
- All automation for each track
- Summary output

### `api_usage.py`
Interact with the API server programmatically.

```bash
# Start server first
python api_server.py

# In another terminal
python examples/api_usage.py
```

Demonstrates:
- Health checking
- Generating tracks via API
- Job status polling
- Getting analytics
- Listing tracks

## Running Examples

1. Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

2. Run any example:
```bash
python examples/basic_workflow.py
```

## Output

All examples output to the `output/` directory:
- `output/audio/` - Generated audio files
- `output/videos/` - Generated videos
- `output/thumbnails/` - Generated thumbnails
- `output/metadata/` - Metadata JSON files

## Integration

To integrate your own music generation model:
1. Edit `orchestrator.py`
2. Find the `generate_music()` method
3. Replace placeholder with your model
4. Run examples to test integration
