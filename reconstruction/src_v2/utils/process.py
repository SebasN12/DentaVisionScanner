from pathlib import Path
import subprocess


def run_command(
    executable: Path | str,
    arguments: list[str | Path | int | float],
    working_directory: Path | str | None = None,
) -> None:
    """
    Execute an external command and stream its output to the terminal.

    Raises:
        FileNotFoundError:
            If the executable does not exist.
        subprocess.CalledProcessError:
            If the process exits with a non-zero return code.
    """

    executable = Path(executable)

    if not executable.exists():
        raise FileNotFoundError(
            f"Executable not found: {executable}"
        )

    command = [
        str(executable),
        *(str(argument) for argument in arguments),
    ]

    print()
    print("Running:")
    print(" ".join(f'"{argument}"' for argument in command))
    print()

    subprocess.run(
        command,
        cwd=working_directory,
        check=True,
    )