# Contributing

Thank you for considering a contribution. This project aims to remain small,
focused and reproducible — please keep PRs scoped accordingly.

## Ground rules

- Open an issue before large changes so we can agree on scope.
- Every new behaviour needs a test.
- Code must pass `ruff`, `mypy` and `pytest` locally.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat: …`, `fix: …`, `docs: …`, `refactor: …`, `test: …`, `ci: …`).

## Setting up a dev environment

```bash
git clone https://github.com/<your-user>/traffic-sign-recognition.git
cd traffic-sign-recognition
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running the quality gates

```bash
ruff check .
ruff format --check .
mypy
pytest
```

CI runs the same commands on every push and pull request — if it passes
locally, it passes on GitHub.

## Notebook hygiene

Jupyter outputs must not be committed. Install the pre-commit hook once:

```bash
nbstripout --install
```

## Reporting security issues

See [`SECURITY.md`](SECURITY.md).

## Licence

By submitting a pull request you agree that your contribution is licensed
under the MIT licence of this repository.
