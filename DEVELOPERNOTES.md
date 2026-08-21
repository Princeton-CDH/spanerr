# Developer Notes

This project uses git flow branching conventions via [git-flow-next](https://github.com/gittower/git-flow-next).

> [!NOTE]
> Make sure you are using the correct version of git flow.
> The original [git-flow](https://github.com/nvie/gitflow) and its successor [git-flow-avh](https://github.com/petervanderdoes/gitflow-avh) are no longer maintained.
> While `git-flow-next` is backwards compatible, this project assumes the workflow and features of `git-flow-next`.

For development, we assume the usage of [uv](https://docs/astral.sh/uv/).
`uv` is compatible with the use of `pip` for python package management and a tool
of your choice for creating python virtual environments (e.g., `mamba`, `venv`).

## Initial setup and installation

### Initialize and configure git-flow in your local repository

Install `git-flow-next` if it's not installed.
It can be installed via Homebrew or manual installation.
See `git-flow-next`'s [installation documentation](https://git-flow.sh/docs/installation/) for more details.

To initialize git-flow run:

```sh
git flow init --preset=classic --defaults
```

This package uses custom configurations options for git-flow including the use of custom git-flow hooks which are defined in `gitflow-hooks`.
Run the provided `setup_gitflow.sh` script to update git-flow's configuration.

```sh
sh setup_gitflow.sh
```

These configuration options are set in the local git config (`.git/config`).

To display an overview of the current git-flow configuration, branch structure, and workflow status run:

```sh
git flow overview
```

### Install uv

Install `uv` if it's not installed.
It can be installed via PyPi, Homebrew, or a standalone installer.
See `uv`'s [installation documentation](https://docs.astral.sh/uv/getting-started/installation)
for more details.

To explicitly sync the project's dependencies, including optional dependencies for
development and testing, to your local environment run:

```sh
uv sync
```

Note that `uv` performs syncing and locking automatically (e.g., any time
`uv run` is invoked). By default, syncing will remove any packages not
specifically specified in the `pyproject.toml`.

### Install pre-commit hooks

Anyone who wants to contribute to this codebase should install the configured pre-commit hooks.

To install pre-commit run:

```sh
uv tool install pre-commit --with pre-commit-uv
```

To install the configure pre-commit hooks run:

```sh
pre-commit install
```

This will configure a pre-commit hooks to automatically lint and format python code with [ruff](https://github.com/astral-sh/ruff) and [black](https://github.com/psf/black).

To run pre-commit explicitly run:

```sh
pre-commit run --all-files
```

Pre-commit hooks and formatting conventions were added at version 0.5, so `git blame` may not reflect the true author of a given change. To make `git blame` more accurate, ignore formatting revisions:

```sh
git blame <FILE> --ignore-revs-file .git-blame-ignore-revs
```

Or configure your git to always ignore styling revision commits:

```sh
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

## Unit testing

Unit tests are set up to be used with [pytest](https://docs.pytest.org/).

To run the tests, run:

```sh
uv run pytest
```

## Pull requests

To propose code changes, create a pull request against the **develop** branch
(per our git flow workflow). Pull requests should include an update to `CHANGELOG.md`
documenting the changes to the project.

Several GitHub Actions are run for pull requests: unit testing, code coverage,
ruff checks, and a changelog check. By default, the changelog check will fail if
the PR does not update `CHANGELOG.md`. For pull requests that do not modify code,
the "no changelog" label can be added to skip this check.
