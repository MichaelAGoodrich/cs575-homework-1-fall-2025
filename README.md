# cs-575-homework-1-fall-2025
Homework 1 CS 575

Michael A. Goodrich <br>
Brigham Young University <br>
January 2025

---

### Clone the repository

Once you've accepted the assignment, clone the repository. I'm assuming you can set up vscode and clone a repository from within vscode.

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
- ipykernel (for Jupyter notebooks)
- scipy.cluster.hierarchy (to show dendrograms for future homework)
- python-louvain (to find communities using the Louvain algorithm for future homework)

On my machine, these packages installed in `.venv/lib/python3.12/site-packages`. Note that the .gitignore file must say that files in the `.venv` directory should be ignored so that you don't try and upload those packages to the git cloud. 

**Warning** Avoid using pygraphviz on a mac. The linking required to code written in C is sensitive to specific mac configurations. It is much more simple to use pydot instead. 

**Select Python Interpreter** Open the command pallette within visual studio code. (One way to do this is to select `View -> CommandPallette`.) Select `Python:Select Interpreter`. Choose the directory with the virtual environment you just created, `Python 3.12.5 ('.venv') ./.venv/bin/pythyon` if on a mac.

**Configure Tests** Open the command pallette within visual studio code. Select `Python:Configure Tests`. Select `pytest`, and then select the `tests` directory. You can then click on the test tube icon in visual studio code to inspect and run the tests. You'll look at the tests in the `homework_0` directory when you step through the Jupyter notebook tutorial, and you'll complete the tests in the `homework_1` directory when you do homework 1.

I sometimes have a hard time getting vscode and pytest to play well together. The work around is to open a terminal in the root project directory and type `pytest tests` or `pytest tests/your_test_folder_name`. 

**Pre-commit** Pre-commit runs the mypy static type checker and ruff. It also fixes common code style issues like removing trailing white space and tabs. The configuration file is `pre-commit-config.yaml`. I'm including this information as a stub for future semesters since I won't be enforcing pre-commit this semester.

---



