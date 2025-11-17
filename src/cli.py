"""Professional CLI for lo-fi music generation using Typer and Rich.

Features:
- Beautiful terminal UI with Rich
- Progress bars and spinners
- Interactive mode
- Subcommands for different operations
- Config management
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich.logging import RichHandler
from rich.prompt import Prompt, Confirm
from rich import print as rprint

from src.generator import LoFiGenerator
from src.model import ConditionedLoFiModel
from src.tokenizer import LoFiTokenizer
from src.audio_processor import LoFiAudioProcessor
from src.quality_scorer import MusicQualityScorer
from src.utils.resource_manager import ResourceManager

# Initialize Typer app and Rich console
app = typer.Typer(
    name="lofi",
    help="🎵 Ultra-Pro Lo-Fi Music Generator - Production AI Music Generation",
    add_completion=False,
)
console = Console()

# Configure logging with Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, console=console)]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def display_banner():
    """Display startup banner."""
    banner = """
[bold cyan]╔══════════════════════════════════════════════╗
║  🎵  Lo-Fi Music Generator - Ultra Pro  🎵  ║
║         AI-Powered Music Production          ║
╚══════════════════════════════════════════════╝[/bold cyan]
"""
    console.print(banner)


@app.command()
def generate(
    tempo: Optional[float] = typer.Option(None, "--tempo", "-t", help="Tempo in BPM (50-200)", min=50, max=200),
    key: Optional[str] = typer.Option(None, "--key", "-k", help="Musical key (e.g., C, Am, F#)"),
    mood: Optional[str] = typer.Option(None, "--mood", "-m", help="Mood (chill, melancholic, upbeat, etc.)"),
    output: Optional[str] = typer.Option("output/generated.mid", "--output", "-o", help="Output MIDI file path"),
    temperature: float = typer.Option(0.9, "--temperature", help="Sampling temperature (0.1-2.0)", min=0.1, max=2.0),
    max_length: int = typer.Option(1024, "--max-length", help="Maximum token length", min=256, max=4096),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for reproducibility"),
    audio: bool = typer.Option(False, "--audio", "-a", help="Generate audio (WAV) file"),
    lofi_effects: bool = typer.Option(True, "--lofi-effects/--no-lofi-effects", help="Apply lo-fi effects to audio"),
    config: str = typer.Option("config.yaml", "--config", "-c", help="Configuration file path"),
    model_path: Optional[str] = typer.Option(None, "--model", help="Path to trained model"),
):
    """Generate a single lo-fi music track."""
    display_banner()

    try:
        # Load config
        cfg = load_config(config)

        # Resource check
        with console.status("[bold green]Checking system resources...") as status:
            rm = ResourceManager()
            resources = rm.check_all_resources()

            if not resources['all_ok']:
                console.print("[yellow]⚠️  Warning: Low system resources detected[/yellow]")

            device = rm.get_optimal_device()
            console.print(f"[green]✓[/green] Using device: [bold]{device}[/bold]")

        # Initialize components
        with console.status("[bold green]Initializing model...") as status:
            tokenizer = LoFiTokenizer(cfg)
            model = ConditionedLoFiModel(cfg, tokenizer.get_vocab_size())

            if model_path:
                model.load(model_path)
                console.print(f"[green]✓[/green] Loaded model from: {model_path}")

            model.to(device)
            generator = LoFiGenerator(model, tokenizer, cfg, device=device)
            console.print("[green]✓[/green] Model initialized")

        # Generate track
        console.print("\n[bold cyan]🎵 Generating track...[/bold cyan]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating tokens...", total=100)

            tokens, metadata = generator.generate_track(
                tempo=tempo,
                key=key,
                mood=mood,
                max_length=max_length,
                temperature=temperature,
                seed=seed,
            )

            progress.update(task, completed=50)
            progress.update(task, description="Calculating quality score...")

            # Quality score
            scorer = MusicQualityScorer(cfg)
            quality_score = scorer.score_midi_tokens(tokens, metadata)

            progress.update(task, completed=75)
            progress.update(task, description="Saving MIDI file...")

            # Save MIDI
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            generator.tokens_to_midi(tokens, str(output_path))

            progress.update(task, completed=100)

        # Display results
        console.print(f"\n[green]✅ Generation complete![/green]")

        table = Table(title="Generated Track Information", show_header=True, header_style="bold magenta")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="yellow")

        table.add_row("Tempo", f"{metadata['tempo']:.1f} BPM")
        table.add_row("Key", metadata['key'])
        table.add_row("Mood", metadata['mood'])
        table.add_row("Tokens", str(metadata['num_tokens']))
        table.add_row("Quality Score", f"{quality_score:.2f}/10")
        table.add_row("MIDI File", str(output_path))

        console.print(table)

        # Generate audio if requested
        if audio:
            console.print("\n[bold cyan]🔊 Processing audio...[/bold cyan]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Converting to audio...", total=100)

                processor = LoFiAudioProcessor(cfg)
                audio_output = output_path.with_suffix('.wav')

                if lofi_effects:
                    progress.update(task, description="Applying lo-fi effects...")
                    result = processor.process_midi_to_lofi(
                        str(output_path),
                        str(output_path.parent),
                        name=output_path.stem,
                        save_lofi=True,
                        save_clean=False,
                    )
                    audio_path = result.get('lofi_wav_path')
                else:
                    progress.update(task, description="Rendering clean audio...")
                    processor.midi_to_wav(str(output_path), str(audio_output))
                    audio_path = str(audio_output)

                progress.update(task, completed=100)

            console.print(f"[green]✅ Audio saved:[/green] {audio_path}")

    except Exception as e:
        console.print(f"[bold red]❌ Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def batch(
    num_tracks: int = typer.Argument(..., help="Number of tracks to generate"),
    output_dir: str = typer.Option("output/batch", "--output-dir", "-o", help="Output directory"),
    prefix: str = typer.Option("lofi_track", "--prefix", "-p", help="Filename prefix"),
    variety: bool = typer.Option(True, "--variety/--no-variety", help="Ensure variety in tempo/key/mood"),
    min_quality: Optional[float] = typer.Option(None, "--min-quality", help="Minimum quality score (0-10)"),
    audio: bool = typer.Option(False, "--audio", "-a", help="Generate audio files"),
    config: str = typer.Option("config.yaml", "--config", "-c", help="Configuration file path"),
    model_path: Optional[str] = typer.Option(None, "--model", help="Path to trained model"),
):
    """Generate multiple tracks in batch mode."""
    display_banner()

    try:
        cfg = load_config(config)

        # Initialize
        with console.status("[bold green]Initializing...") as status:
            rm = ResourceManager()
            device = rm.get_optimal_device()

            tokenizer = LoFiTokenizer(cfg)
            model = ConditionedLoFiModel(cfg, tokenizer.get_vocab_size())
            if model_path:
                model.load(model_path)
            model.to(device)

            generator = LoFiGenerator(model, tokenizer, cfg, device=device)
            scorer = MusicQualityScorer(cfg)
            if audio:
                processor = LoFiAudioProcessor(cfg)

        console.print(f"\n[bold cyan]🎵 Generating {num_tracks} tracks...[/bold cyan]\n")

        # Generate tracks
        high_quality_count = 0
        output_path = Path(output_dir)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            batch_task = progress.add_task("Batch generation...", total=num_tracks)

            metadata_list = generator.batch_generate(
                num_tracks=num_tracks,
                output_dir=str(output_path),
                name_prefix=prefix,
                ensure_variety=variety,
            )

            for i, meta in enumerate(metadata_list):
                progress.update(batch_task, description=f"Track {i+1}/{num_tracks} - Scoring quality...")

                # Load and score
                if 'output_path' in meta:
                    tokens_data = tokenizer.tokenize_midi(meta['output_path'], check_quality=False)
                    if tokens_data:
                        quality_score = scorer.score_midi_tokens(tokens_data['tokens'], meta)
                        meta['quality_score'] = quality_score

                        if min_quality is None or quality_score >= min_quality:
                            high_quality_count += 1

                            # Process audio if requested
                            if audio:
                                progress.update(batch_task, description=f"Track {i+1}/{num_tracks} - Processing audio...")
                                processor.process_midi_to_lofi(
                                    meta['output_path'],
                                    str(output_path),
                                    name=f"{prefix}_{i+1:03d}",
                                    save_lofi=True,
                                    save_clean=False,
                                )

                progress.update(batch_task, advance=1)

        # Results
        console.print(f"\n[green]✅ Batch generation complete![/green]")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")

        table.add_row("Total Tracks", str(num_tracks))
        table.add_row("High Quality", str(high_quality_count))
        table.add_row("Success Rate", f"{high_quality_count/num_tracks*100:.1f}%")
        table.add_row("Output Directory", str(output_path))

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]❌ Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def interactive():
    """Interactive mode for music generation."""
    display_banner()

    console.print("\n[bold cyan]🎹 Interactive Generation Mode[/bold cyan]\n")

    try:
        # Load config
        config_path = Prompt.ask("Configuration file", default="config.yaml")
        cfg = load_config(config_path)

        # Initialize
        with console.status("[bold green]Initializing..."):
            rm = ResourceManager()
            device = rm.get_optimal_device()

            tokenizer = LoFiTokenizer(cfg)
            model = ConditionedLoFiModel(cfg, tokenizer.get_vocab_size())
            model.to(device)

            generator = LoFiGenerator(model, tokenizer, cfg, device=device)
            scorer = MusicQualityScorer(cfg)

        console.print("[green]✓[/green] Ready to generate!\n")

        while True:
            # Get parameters
            tempo = Prompt.ask("Tempo (BPM)", default="75")
            key = Prompt.ask("Key", default="Am")
            mood = Prompt.ask("Mood", default="chill", choices=["chill", "melancholic", "upbeat", "relaxed", "dreamy"])
            output = Prompt.ask("Output file", default="output/interactive.mid")

            # Generate
            console.print("\n[bold cyan]🎵 Generating...[/bold cyan]")

            tokens, metadata = generator.generate_track(
                tempo=float(tempo),
                key=key,
                mood=mood,
            )

            quality_score = scorer.score_midi_tokens(tokens, metadata)

            # Save
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            generator.tokens_to_midi(tokens, str(output_path))

            console.print(f"\n[green]✅ Generated![/green] Quality: [yellow]{quality_score:.2f}/10[/yellow]")
            console.print(f"Saved to: {output_path}\n")

            # Continue?
            if not Confirm.ask("Generate another track?", default=True):
                break

        console.print("\n[bold green]👋 Thanks for using Lo-Fi Generator![/bold green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[bold red]❌ Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def info(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Configuration file path"),
):
    """Display system and model information."""
    display_banner()

    try:
        cfg = load_config(config)
        rm = ResourceManager()

        # System info
        console.print("\n[bold cyan]💻 System Information[/bold cyan]\n")

        resources = rm.check_all_resources()

        sys_table = Table(show_header=True, header_style="bold magenta")
        sys_table.add_column("Resource", style="cyan")
        sys_table.add_column("Status", style="yellow")
        sys_table.add_column("Details", style="white")

        # Device
        device = rm.get_optimal_device()
        sys_table.add_row(
            "Device",
            "[green]✓[/green]" if resources['all_ok'] else "[yellow]⚠[/yellow]",
            device.upper()
        )

        # GPU
        if resources['gpu']['available']:
            gpu_info = resources['gpu']['info']
            gpu_status = f"{len(gpu_info['devices'])} GPU(s) - {gpu_info['free_memory_gb']:.1f}GB free"
        else:
            gpu_status = "Not available"

        sys_table.add_row(
            "GPU",
            "[green]✓[/green]" if resources['gpu']['available'] else "[red]✗[/red]",
            gpu_status
        )

        # Memory
        mem_info = resources['memory']['info']
        sys_table.add_row(
            "Memory",
            "[green]✓[/green]" if resources['memory']['ok'] else "[yellow]⚠[/yellow]",
            f"{mem_info['available_gb']:.1f}GB / {mem_info['total_gb']:.1f}GB"
        )

        # Disk
        disk_info = resources['disk']['info']
        sys_table.add_row(
            "Disk Space",
            "[green]✓[/green]" if resources['disk']['ok'] else "[yellow]⚠[/yellow]",
            f"{disk_info['free_gb']:.1f}GB / {disk_info['total_gb']:.1f}GB"
        )

        console.print(sys_table)

        # Model info
        console.print("\n[bold cyan]🎵 Model Configuration[/bold cyan]\n")

        model_table = Table(show_header=True, header_style="bold magenta")
        model_table.add_column("Parameter", style="cyan")
        model_table.add_column("Value", style="yellow")

        model_cfg = cfg['model']
        model_table.add_row("Embedding Dimension", str(model_cfg['embedding_dim']))
        model_table.add_row("Layers", str(model_cfg['num_layers']))
        model_table.add_row("Attention Heads", str(model_cfg['num_heads']))
        model_table.add_row("Context Length", str(model_cfg['context_length']))
        model_table.add_row("Total Parameters", "~117M")

        console.print(model_table)

    except Exception as e:
        console.print(f"[bold red]❌ Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    config: str = typer.Option("config.yaml", "--config", "-c", help="Configuration file path"),
    model_path: Optional[str] = typer.Option(None, "--model", help="Path to trained model"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
):
    """Start the production API server."""
    display_banner()

    console.print(f"\n[bold cyan]🚀 Starting API server on {host}:{port}[/bold cyan]\n")

    try:
        from src.api import LoFiAPI

        cfg = load_config(config)
        api = LoFiAPI(cfg, model_path)

        console.print("[green]✓[/green] API server initialized")
        console.print(f"[cyan]➜[/cyan] API docs: http://{host}:{port}/docs")
        console.print(f"[cyan]➜[/cyan] Health check: http://{host}:{port}/api/v1/health\n")

        api.run(host=host, port=port, reload=reload)

    except Exception as e:
        console.print(f"[bold red]❌ Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def version():
    """Display version information."""
    console.print("\n[bold cyan]Lo-Fi Music Generator[/bold cyan]")
    console.print("Version: [yellow]2.0.0[/yellow]")
    console.print("Status: [green]Ultra-Pro Edition[/green]\n")


if __name__ == "__main__":
    app()
