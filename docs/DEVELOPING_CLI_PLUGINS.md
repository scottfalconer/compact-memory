# Developing CLI Plugins for Gist Memory

Gist Memory supports extending its Command Line Interface (CLI) through plugins. This allows you to add new commands or command groups to the `gist-memory` executable.

## How CLI Plugins Work

CLI plugins are discovered using Python's entry points mechanism, specifically the `gist_memory.cli` group.

When Gist Memory starts, it looks for installed packages that declare entry points in this group. For each discovered plugin, it attempts to load it.

## Plugin Structure

An entry point for a CLI plugin should point to one of two things:

1.  **A `typer.Typer` instance:**
    If your entry point loads a `typer.Typer` object, this object will be added as a new top-level command group to the main `gist-memory` app. The name of the command group will be taken from the entry point's name.

    *Example `pyproject.toml`:*
    ```toml
    [project.entry-points."gist_memory.cli"]
    myplugingroup = "my_plugin_package.cli:my_group_app"
    ```

    *Example `my_plugin_package/cli.py`:*
    ```python
    import typer

    my_group_app = typer.Typer(help="My awesome plugin command group!")

    @my_group_app.command("hello")
    def hello_world(name: str = "World"):
        typer.echo(f"Hello {name} from my plugin!")

    # To use: gist-memory myplugingroup hello --name "Plugin User"
    ```

2.  **A callable (function):**
    If your entry point loads a callable (e.g., a function), this function will be called with the main `gist-memory` `typer.Typer` application instance as its argument. Your function can then directly add commands or sub-groups to this main application.

    *Example `pyproject.toml`:*
    ```toml
    [project.entry-points."gist_memory.cli"]
    myplugincommands = "my_plugin_package.cli:register_my_commands"
    ```

    *Example `my_plugin_package/cli.py`:*
    ```python
    import typer

    def register_my_commands(app: typer.Typer):
        @app.command("plugin-greet")
        def greet_from_plugin(name: str = "Plugin User"):
            typer.echo(f"Greetings {name}, from a direct plugin command!")

    # To use: gist-memory plugin-greet --name "Direct User"
    ```

## Current Capabilities and Targeting Command Groups

As of the current version, the plugin system directly supports adding new **top-level command groups** (via returning a `Typer` instance) or **top-level commands** (via a callable that adds to the main `app`).

The Gist Memory CLI has internal command groups like `agent`, `config`, and `dev`. **Currently, there is no direct mechanism for a plugin to register its commands or groups to appear *underneath* these pre-defined Gist Memory groups (e.g., to add a command as `gist-memory dev my-plugin-tool`).**

**Considerations for Plugin Developers:**

*   **Naming:** Choose clear and unique names for your plugin's command groups or commands to avoid clashes with built-in commands or other plugins.
*   **Functionality:** If your plugin provides functionality that is conceptually tied to agent operations, development tools, or configuration, you should clearly state this in your plugin's documentation.
*   **Future Enhancements (Potential):** The Gist Memory plugin system might be enhanced in the future to allow plugins to specify a target group (like `dev` or `agent`). If this capability is added, plugin authors would need to adapt their registration process. For now, all plugin-contributed commands will appear at the top level of the `gist-memory` CLI.

## Best Practices

*   Keep your plugin's CLI focused and well-documented.
*   Provide clear help texts for all commands, arguments, and options your plugin adds.
*   Manage dependencies carefully if your plugin requires additional packages.

By following these guidelines, you can create powerful CLI extensions for Gist Memory.
```
