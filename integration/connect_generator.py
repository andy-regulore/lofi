"""
Integration layer connecting original LoFi music generator to automation orchestrator.

This module bridges the gap between:
- Original: GPT-2 based MIDI generation (src/generator.py)
- New: Complete automation suite (orchestrator.py, api_server.py, etc.)

Usage:
    from integration.connect_generator import get_generator

    generator = get_generator()
    result = generator.generate(mood='chill', duration=180)

Author: Claude
License: MIT
