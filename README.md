# cs-575-in-class-exercises
Code for in-class exercises in CS 575

Michael A. Goodrich <br>
Brigham Young University <br>
December 2024 -- April 2025

---

### Before Beginning

**Virtual Environment.**
All in-class code is intended to be run within an virtual environment. Execute the following:
- Open a terminal in vscode
- Deactivate conda if you are running it
- Install venv in the .venv directory from the project's root directory: `python3 -m venv .venv`
- Activate the virtual environment. 
  - If on mac, this is done by: `source .venv/bin/activate`
  - If on windows, this is done by `myenv\Scripts\activate`

You'll know this has worked if you see the command line prompt preceded by "(.venv)". Note that you must deactivate `conda` if you are running it before activating your virtual environment.

To deactivate your virtual environment, type `deactivate`.

**Create directories** Create a *src* directory and a *tests* directory
- `mkdir src`
- `mkdir tests`

**Dependencies** This project has both required and optional dependencies. Both are specified in the file `pyproject.toml`. The optional dependencies aren't required for your code to run, but they are useful for doing good development. Run the following command from an integrated terminal: 

`pip install --editable ".[dev]"` 

This command says to install the optional dependencies used in the **dev**elopment stage. The `--editable` flag means that you won't need to install the dependencies when you make changes and is appropriate for the development stage of the code.

The optional dependencies are
- pytest (unit and integration test environment)
- mypy (static type checker)

The mandatory dependencies are
- pydot
- networkx
- types-networkx (to enable networkx type hints)
- numpy
- matplotlib (pyplot)
- scipy.cluster.hierarchy (to show dendrograms)
- python-louvain (to find communities using the Louvain algorithm)

On my machine, these packages installed in `.venv/lib/python3.12/site-packages`. Note that the .gitignore file must say that files in the `.venv` directory should be ignored so that you don't try and upload those packages to the git cloud. 

**Warning:** Avoid using pygraphviz on a mac. The linking required to code written in C doesn't seem to work. Use pydot instead. 

**Pre-commit.** Pre-commit runs the mypy static type checker and ruff. It also fixes common code style issues like removing trailing white space and tabs. The configuration file is `pre-commit-config.yaml`

