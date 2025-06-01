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

Use `gist-memory package create` to generate a template. Validate your package
with `gist-memory package validate <path>`.

Run an experiment from a package with:

```
gist-memory experiment run-package /path/to/package --experiment experiments/example.yaml
```
