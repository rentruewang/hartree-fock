import rich
from hydra import main
from omegaconf import DictConfig

from . import hartree_fork
from .hartree_fork import HFInput


@main(version_base=None, config_path="conf", config_name="main")
def run(cfg: DictConfig):
    rich.print(cfg)

    molecule = cfg["molecule"]
    hf_input = HFInput.parse(molecule)
    energy = hartree_fork.run(hf_input)

    rich.print(f"Hartree Fock energy: {energy}")


if __name__ == "__main__":
    run()
