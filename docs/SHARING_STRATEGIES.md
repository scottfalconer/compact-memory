# Sharing Compression Strategies

This document describes the experimental "strategy package" format.

A package is a directory containing a `strategy_package.yaml` manifest, the
strategy implementation, optional experiment configurations and supporting
files. Packages make it easier to share and run custom strategies.

```
my_strategy/
    strategy.py                 # implementation
    strategy_package.yaml       # manifest
    requirements.txt            # optional dependencies
    README.md                   # description
    experiments/
        example.yaml            # optional experiment config
```

The manifest has a few required fields:

```yaml
package_format_version: "1.0"
strategy_id: my-strategy
strategy_class_name: MyStrategy
strategy_module: strategy
display_name: My Strategy
version: "0.1.0"
authors:
  - You
description: Short description.
```

Use `compact-memory package create` to generate a template. Validate your package
with `compact-memory package validate <path>`.

Run an experiment from a package with:

```
compact-memory experiment run-package /path/to/package --experiment experiments/example.yaml
```

## Using Strategy Packages as Plugins

Installable Python packages can register a strategy via the
`compact_memory.strategies` entry point group. A minimal `pyproject.toml` snippet:

```toml
[project.entry-points."compact_memory.strategies"]
my_strategy = "my_package.module:MyStrategy"
```

When such a package is installed, the strategy is automatically discovered on
startup.

For local experimentation, place a strategy package directory inside
`$COMPACT_MEMORY_PLUGINS_PATH` or the default user plugin directory
`$(platformdirs.user_data_dir('compact_memory','CompactMemoryTeam'))/plugins`. Multiple
paths may be provided in `COMPACT_MEMORY_PLUGINS_PATH` separated by the system path
separator. Paths listed first take precedence.

The override order is:

1. Built-in strategies
2. Entry point plugins
3. Local plugin directories (highest precedence)

Use `compact-memory strategy list` to verify which strategy implementation is active
and its source.

**Security Warning:** Loading plugins executes arbitrary code. Only install or
place plugins from sources you trust.
