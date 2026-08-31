"""Model downloading functionality with progress tracking."""

import sys
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from fiable.config import settings
from fiable.config.settings import ModelConfig, CACHE_DIR, MODELS
from fiable.utils import helpers


console = Console()


@dataclass
class DownloadResult:
    """Result of downloading a model."""
    model_name: str
    success: bool
    path: Optional[Path] = None
    error: Optional[str] = None
    already_exists: bool = False


def download_model(model: ModelConfig, force: bool = False) -> DownloadResult:
    """
    Download a single model from Hugging Face.
    
    Args:
        model: Model configuration
        force: Force re-download even if exists
        
    Returns:
        DownloadResult with success status
    """
    model_dir = CACHE_DIR / model.local_dir
    
    # Check if already exists
    if model_dir.exists() and not force:
        console.print(f"[yellow]Model '{model.name}' already exists at {model_dir}[/yellow]")
        return DownloadResult(
            model_name=model.name,
            success=True,
            path=model_dir,
            already_exists=True,
        )
    
    console.print(f"[cyan]Downloading {model.name} from Hugging Face...[/cyan]")
    console.print(f"[dim]Repository: {model.repo}[/dim]")
    
    try:
        from huggingface_hub import snapshot_download
        
        # Download with progress
        downloaded_path = snapshot_download(
            repo_id=model.repo,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        
        console.print(f"[green]✓ Downloaded {model.name} successfully[/green]")
        return DownloadResult(
            model_name=model.name,
            success=True,
            path=Path(downloaded_path),
        )
        
    except Exception as e:
        error_msg = str(e)
        console.print(f"[red]✗ Failed to download {model.name}[/red]")
        console.print(f"[red]Error: {error_msg}[/red]")
        
        # Check if it's an authentication error
        if "401" in error_msg or "403" in error_msg:
            console.print("\n[yellow]Authentication required:[/yellow]")
            console.print(f"1. Accept the license at https://huggingface.co/{model.repo}")
            console.print("2. Set your HF token: export HF_TOKEN=your_token_here")
            console.print("   or use: huggingface-cli login")
        
        return DownloadResult(
            model_name=model.name,
            success=False,
            error=error_msg,
        )


def download_models(
    model_names: Optional[List[str]] = None,
    force: bool = False,
) -> List[DownloadResult]:
    """
    Download multiple models.
    
    Args:
        model_names: List of model names to download. If None, download all.
        force: Force re-download even if exists
        
    Returns:
        List of DownloadResult objects
    """
    # Determine which models to download
    if model_names:
        models_to_download = []
        for name in model_names:
            try:
                model = get_model_by_name(name)
                models_to_download.append(model)
            except ValueError as e:
                console.print(f"[red]{e}[/red]")
                continue
    else:
        models_to_download = MODELS
    
    if not models_to_download:
        console.print("[red]No valid models to download[/red]")
        return []
    
    # Ensure base directory exists
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    console.print(f"\n[bold]Downloading {len(models_to_download)} model(s)...[/bold]\n")
    
    results = []
    for model in models_to_download:
        result = download_model(model, force=force)
        results.append(result)
        console.print()  # Blank line between models
    
    # Summary
    success_count = sum(1 for r in results if r.success)
    already_exist = sum(1 for r in results if r.already_exists)
    failed_count = len(results) - success_count
    
    console.print("[bold]Download Summary:[/bold]")
    console.print(f"  Successful: {success_count}")
    if already_exist:
        console.print(f"  Already existed: {already_exist}")
    if failed_count > 0:
        console.print(f"  [red]Failed: {failed_count}[/red]")
    
    return results


def list_downloaded_models() -> List[ModelConfig]:
    """List all downloaded models."""
    downloaded = []
    for model in MODELS:
        model_dir = CACHE_DIR / model.local_dir
        if model_dir.exists():
            downloaded.append(model)
    return downloaded
