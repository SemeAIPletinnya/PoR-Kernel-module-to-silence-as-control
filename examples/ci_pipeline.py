"""Example CI pipeline invocation for the PoR CLI."""

from pathlib import Path

from por_kernel.cli import run


def main() -> None:
    config_path = Path("config.yaml")
    run(config_path)


if __name__ == "__main__":
    main()
