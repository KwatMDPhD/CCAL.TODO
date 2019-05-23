from .run_command import run_command


def install_python_libraries(libraries):

    libraries_installed = tuple(
        line.split()[0]
        for line in run_command("pip list").stdout.strip().split(sep="\n")[2:]
    )

    for library in libraries:

        if library not in libraries_installed:

            run_command(f"pip install {library}")

        else:

            print(f"{library} is already installed.")
