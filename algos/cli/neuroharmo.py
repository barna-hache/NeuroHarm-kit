#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path
import click
import os

# Base directory containing algorithm folders
BASEDIR = Path("/opt/toolkit/algos")


def run_algo(algo_name, input_image, output_dir, **kwargs):
    """
    Launch the harmonization script for a given algorithm, ensuring the correct conda env PATH.
    """
    # Path to the conda environment and its python
    env_name = f"{algo_name.lower()}_env"
    env_prefix = f"/opt/conda/envs/{env_name}"
    if algo_name.lower() == 'stgan':
        env_prefix = "/opt/conda/envs/haca3_env"
    env_python = Path(env_prefix) / "bin/python"

    # Ensure subprocesses find binaries (e.g., N4BiasFieldCorrection) in the env
    os.environ["PATH"] = str(Path(env_prefix) / "bin") + ":" + os.environ.get("PATH", "")

    # Construct script path
    script = BASEDIR / algo_name / f"harmonize_with_{algo_name.lower()}.py"

    cmd = [str(env_python), str(script), input_image, "--output_dir", output_dir]

    # Add additional keyword arguments
    for key, val in kwargs.items():
        flag = f"--{key}"
        if isinstance(val, bool):
            cmd.extend([flag, str(val)])
        elif isinstance(val, list):
            cmd.append(flag)
            cmd.extend(str(v) for v in val)
        else:
            cmd.extend([flag, str(val)])

    # Debug print (optional)
    print(f"Running command: {' '.join(cmd)}")

    # Execute the command
    return subprocess.call(cmd)


@click.group()
def cli():
    """neuroharmo_toolkit: Harmonize MRI images using various algorithms."""
    pass

@cli.command()
@click.argument("input_image", type=click.Path(exists=True))
@click.option("--apply_preproc_steps", type=bool, default=True)
@click.option("--output_dir", default="output")
@click.option("--save_preprocess", type=bool, default=False)
def disarmpp(input_image, apply_preproc_steps, output_dir, save_preprocess):
    """Run DISARMpp harmonization"""
    sys.exit(run_algo(
        "DISARMpp", input_image, output_dir,
        apply_preproc_steps=apply_preproc_steps,
        save_preprocess=save_preprocess
    ))

@cli.command()
@click.argument("input_image", type=click.Path(exists=True))
@click.option("--apply_preproc_steps", type=bool, default=True)
@click.option("--theta", multiple=True, type=float, default=[10.0, 20.0])
@click.option("--output_dir", default="output")
@click.option("--save_preprocess", type=bool, default=False)
def haca3(input_image, apply_preproc_steps, theta, output_dir, save_preprocess):
    """Run HACA3 harmonization"""
    sys.exit(run_algo(
        "HACA3", input_image, output_dir,
        apply_preproc_steps=apply_preproc_steps,
        theta=list(theta),
        save_preprocess=save_preprocess
    ))

@cli.command()
@click.argument("input_image", type=click.Path(exists=True))
@click.option("--apply_preproc_steps", type=bool, default=True)
@click.option("--output_dir", default="output")
@click.option("--save_preprocess", type=bool, default=False)
def iguane(input_image, apply_preproc_steps, output_dir, save_preprocess):
    """Run IGUANE harmonization"""
    sys.exit(run_algo(
        "IGUANE", input_image, output_dir,
        apply_preproc_steps=apply_preproc_steps,
        save_preprocess=save_preprocess
    ))

@cli.command()
@click.argument("input_image", type=click.Path(exists=True))
@click.option("--output_dir", default="output")
@click.option("--n_axial_slices", type=int, default=60)
@click.option("--save_preprocess", type=bool, default=False)
def murd(input_image, output_dir, n_axial_slices, save_preprocess):
    """Run MURD harmonization"""
    sys.exit(run_algo(
        "MURD", input_image, output_dir,
        n_axial_slices=n_axial_slices,
        save_preprocess=save_preprocess
    ))

@cli.command()
@click.argument("input_image", type=click.Path(exists=True))
@click.option("--apply_preproc_steps", type=bool, default=True)
@click.option("--output_dir", default="output")
@click.option("--save_preprocess", type=bool, default=False)
def stgan(input_image, apply_preproc_steps, output_dir, save_preprocess):
    """Run STGAN harmonization"""
    sys.exit(run_algo(
        "STGAN", input_image, output_dir,
        apply_preproc_steps=apply_preproc_steps,
        save_preprocess=save_preprocess
    ))

if __name__ == "__main__":
    cli()
