"""Model quantization functionality with metadata tracking."""

import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from fiable.config import settings
from fiable.config.settings import (
    ModelConfig,
    CACHE_DIR,
    OUTPUT_DIR,
    LLAMA_QUANTIZE,
    MODELS,
    QUANT_TYPES,
    get_model_by_name,
    get_fp16_path,
    get_quantized_path,
)
from fiable.utils import helpers


console = Console()


@dataclass
class QuantizationResult:
    """Result of quantizing a model."""
    model_name: str
    quantization_type: str
    success: bool
    output_path: Optional[Path] = None
    file_size_gb: float = 0.0
    duration_seconds: float = 0.0
    error: Optional[str] = None
    already_exists: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = asdict(self)
        if self.output_path:
            data['output_path'] = str(self.output_path)
        return data


def convert_to_gguf(model: ModelConfig, force: bool = False) -> Tuple[bool, Optional[Path]]:
    """
    Convert HuggingFace model to GGUF FP16 format.
    
    Args:
        model: Model configuration
        force: Force reconversion even if exists
        
    Returns:
        Tuple of (success, fp16_path)
    """
    fp16_path = get_fp16_path(model)
    
    # Check if already exists
    if fp16_path.exists() and not force:
        console.print(f"[yellow]FP16 GGUF already exists: {fp16_path}[/yellow]")
        return True, fp16_path
    
    console.print(f"[cyan]Converting {model.name} to GGUF FP16...[/cyan]")
    
    # Clone llama.cpp repo if needed
    llama_cpp_dir = CACHE_DIR / "llama.cpp"
    if not llama_cpp_dir.exists():
        console.print("[dim]Cloning llama.cpp repository...[/dim]")
        try:
            helpers.run_command(
                f"git clone https://github.com/ggerganov/llama.cpp.git {llama_cpp_dir}"
            )
            helpers.run_command(f"pip install -q -r {llama_cpp_dir}/requirements.txt")
        except Exception as e:
            console.print(f"[red]Failed to clone llama.cpp: {e}[/red]")
            return False, None
    
    # Convert model
    model_dir = CACHE_DIR / model.local_dir
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    
    if not model_dir.exists():
        console.print(f"[red]Model directory not found: {model_dir}[/red]")
        console.print("[yellow]Run download first[/yellow]")
        return False, None
    
    try:
        cmd = f"python3 {convert_script} {model_dir} --outfile {fp16_path} --outtype f16"
        helpers.run_command(cmd)
        console.print(f"[green]✓ Converted to FP16 GGUF: {fp16_path}[/green]")
        return True, fp16_path
    except Exception as e:
        console.print(f"[red]Conversion failed: {e}[/red]")
        return False, None


def quantize_model(
    model: ModelConfig,
    fp16_path: Path,
    quant_type: str,
    force: bool = False,
) -> QuantizationResult:
    """
    Quantize a single model to a specific quantization type.
    
    Args:
        model: Model configuration
        fp16_path: Path to FP16 GGUF file
        quant_type: Quantization type (Q4_K_M, etc.)
        force: Force re-quantization even if exists
        
    Returns:
        QuantizationResult with metadata
    """
    output_path = get_quantized_path(model, quant_type)
    
    # Check if already exists
    if output_path.exists() and not force:
        file_size = helpers.get_file_size_gb(output_path)
        console.print(f"[yellow]Skipping {quant_type} - already exists ({file_size:.2f} GB)[/yellow]")
        return QuantizationResult(
            model_name=model.name,
            quantization_type=quant_type,
            success=True,
            output_path=output_path,
            file_size_gb=file_size,
            already_exists=True,
        )
    
    console.print(f"[cyan]Creating {quant_type} quantization...[/cyan]")
    
    start_time = time.time()
    
    try:
        cmd = f"{LLAMA_QUANTIZE} {fp16_path} {output_path} {quant_type}"
        result = helpers.run_command(cmd, check=False)
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            file_size = helpers.get_file_size_gb(output_path)
            console.print(
                f"[green]✓ Created {quant_type} ({file_size:.2f} GB) "
                f"in {helpers.format_duration(duration)}[/green]"
            )
            return QuantizationResult(
                model_name=model.name,
                quantization_type=quant_type,
                success=True,
                output_path=output_path,
                file_size_gb=file_size,
                duration_seconds=duration,
            )
        else:
            console.print(f"[red]✗ Failed to create {quant_type}[/red]")
            return QuantizationResult(
                model_name=model.name,
                quantization_type=quant_type,
                success=False,
                error=result.stderr,
                duration_seconds=duration,
            )
            
    except Exception as e:
        duration = time.time() - start_time
        console.print(f"[red]✗ Error creating {quant_type}: {e}[/red]")
        return QuantizationResult(
            model_name=model.name,
            quantization_type=quant_type,
            success=False,
            error=str(e),
            duration_seconds=duration,
        )


def quantize_models(
    model_names: Optional[List[str]] = None,
    quant_types: Optional[List[str]] = None,
    force: bool = False,
) -> List[QuantizationResult]:
    """
    Quantize multiple models to multiple quantization types.
    
    Args:
        model_names: List of model names. If None, quantize all.
        quant_types: List of quantization types. If None, use all.
        force: Force re-quantization even if exists
        
    Returns:
        List of QuantizationResult objects
    """
    # Determine which models to quantize
    if model_names:
        models_to_quantize = []
        for name in model_names:
            try:
                model = get_model_by_name(name)
                models_to_quantize.append(model)
            except ValueError as e:
                console.print(f"[red]{e}[/red]")
                continue
    else:
        models_to_quantize = MODELS
    
    # Determine quantization types
    if quant_types is None:
        quant_types = QUANT_TYPES
    
    if not models_to_quantize:
        console.print("[red]No valid models to quantize[/red]")
        return []
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    console.print(
        f"\n[bold]Quantizing {len(models_to_quantize)} model(s) "
        f"to {len(quant_types)} type(s)...[/bold]\n"
    )
    
    all_results = []
    
    for model in models_to_quantize:
        console.print(f"\n[bold cyan]Processing {model.name}...[/bold cyan]")
        
        # Convert to FP16 GGUF first
        success, fp16_path = convert_to_gguf(model, force=force)
        if not success or not fp16_path:
            console.print(f"[red]Skipping {model.name} - conversion failed[/red]")
            continue
        
        # Quantize to each type
        for quant_type in quant_types:
            result = quantize_model(model, fp16_path, quant_type, force=force)
            all_results.append(result)
        
        console.print()  # Blank line between models
    
    # Summary
    success_count = sum(1 for r in all_results if r.success)
    already_exist = sum(1 for r in all_results if r.already_exists)
    failed_count = len(all_results) - success_count
    
    console.print("[bold]Quantization Summary:[/bold]")
    console.print(f"  Successful: {success_count}")
    if already_exist:
        console.print(f"  Already existed: {already_exist}")
    if failed_count > 0:
        console.print(f"  [red]Failed: {failed_count}[/red]")
    
    # Calculate total size
    total_size = sum(r.file_size_gb for r in all_results if r.success)
    console.print(f"  Total size: {total_size:.2f} GB")
    
    return all_results


def list_quantized_models() -> List[Path]:
    """List quantized GGUF models in cache (excludes FP16/BF16 sources)."""
    if not OUTPUT_DIR.exists():
        return []
    from fiable.core.metrics import is_baseline_quant, parse_model_identity

    out = []
    for path in OUTPUT_DIR.glob("*.gguf"):
        _name, quant = parse_model_identity(path)
        if not is_baseline_quant(quant):
            out.append(path)
    return out
