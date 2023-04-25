import typing
from pathlib import Path

import rich
from hydra import main
from omegaconf import DictConfig, OmegaConf

import hartree_fork
from hartree_fork import HFInput, paths


def make_yaml(fname: str):
    if not fname.endswith(".yaml"):
        fname = f"{fname}.yaml"
    return fname


@main(version_base=None, config_path=paths.CONF, config_name="main")
def run(cfg: DictConfig):
    molecule = cfg["molecule"]
    cfg = typing.cast(
        DictConfig, OmegaConf.load(Path(paths.CONF) / make_yaml(cfg["molecule"]))
    )

    rich.print(cfg)

    hf_input = HFInput.from_config(molecule, cfg)
    energy = hartree_fork.run(hf_input)

    rich.print(f"Hartree Fock energy: {energy} (Hartree)")


if __name__ == "__main__":
    run()
