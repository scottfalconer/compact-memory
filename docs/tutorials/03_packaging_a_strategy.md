# Tutorial 3: From Idea to Plugin - Packaging Your Compact Memory Strategy
This tutorial guides you through the process of packaging a custom compression strategy you've developed for Compact Memory. Packaging makes your strategy easily shareable, discoverable, and installable by others. We'll use the \`dev create-strategy-package\` CLI command and prepare the strategy for distribution.
## Prerequisites
*   `Compact Memory installed (\`pip install compact-memory\`).`
*   `You have already developed a custom compression strategy (e.g., as a \`.py\` file). For this tutorial, we'll assume you have a strategy in \`my_awesome_strategy.py\` containing a class \`MyAwesomeStrategy\` which inherits from \`CompressionStrategy\`.`
*   `Your \`MyAwesomeStrategy\` class has a unique string attribute \`id = "awesome_strat"\`.`
## Goal
By the end of this tutorial, you will have a well-structured strategy package directory ready for sharing, either directly or as part of a larger Python package.
## Steps
### 1. Generate the Package Template
Compact Memory's CLI provides a handy tool to bootstrap a strategy package. Open your terminal and navigate to where you want to create your package directory.
```bash
compact-memory dev create-strategy-package --name MyAwesomeStrategyPackage
```
This command will create a new directory named \`MyAwesomeStrategyPackage\` with the following structure:
```text
MyAwesomeStrategyPackage/
├── strategy.py                      # Template strategy implementation
├── strategy_package.yaml            # Manifest file
├── README.md                        # Basic README for your package
├── requirements.txt                 # For dependencies
└── experiments/                     # For example experiments
    └── example.yaml
```
### 2. Add Your Strategy Code
Now, replace the contents of the template \`MyAwesomeStrategyPackage/strategy.py\` with your actual strategy code from \`my_awesome_strategy.py\`.
Let's assume your \`my_awesome_strategy.py\` looked something like this:
```python
from compact_memory.compression.strategies_abc import CompressionStrategy, CompressedMemory
from compact_memory.compression.trace import CompressionTrace
# ... any other imports your strategy needs ...

class MyAwesomeStrategy(CompressionStrategy):
    id = "awesome_strat" # Your unique strategy ID

    def __init__(self, custom_param: int = 5):
        self.custom_param = custom_param
        # ...

    def compress(self, text_or_chunks, llm_token_budget, **kwargs):
        # Your brilliant compression logic here
        compressed_text = f"Compressed with custom_param {self.custom_param}: {str(text_or_chunks)[:100]}"
        trace = CompressionTrace(
            strategy_name=self.id,
            strategy_params={'custom_param': self.custom_param},
            input_summary={'original_length': len(str(text_or_chunks))},
            output_summary={'compressed_length': len(compressed_text)},
            final_compressed_object_preview=compressed_text[:50]
        )
        return CompressedMemory(text=compressed_text), trace
```
Copy this class into \`MyAwesomeStrategyPackage/strategy.py\`. You can delete the template \`MyStrategy\` class that was there.
### 3. Update the Manifest File
The \`strategy_package.yaml\` file is crucial. It tells Compact Memory about your strategy. Open \`MyAwesomeStrategyPackage/strategy_package.yaml\` and edit it. The template will look like this:
```yaml
package_format_version: "1.0"
strategy_id: MyAwesomeStrategyPackage # Placeholder from command
strategy_class_name: MyStrategy         # Placeholder from command
strategy_module: strategy
display_name: MyAwesomeStrategyPackage  # Placeholder from command
version: "0.1.0"
authors: []
description: Describe the strategy
```
Modify it to accurately reflect your strategy. For our \`MyAwesomeStrategy\`, it should become:
```yaml
package_format_version: "1.0"
strategy_id: awesome_strat          # Must match the 'id' in your class
strategy_class_name: MyAwesomeStrategy # Your actual class name
strategy_module: strategy             # The name of the .py file (without extension)
display_name: "My Awesome Strategy"   # A user-friendly name
version: "1.0.0"                      # Your strategy's version
authors:
  - "Your Name <your.email@example.com>"
description: "An awesome strategy that demonstrates packaging."
# Optional: Add dependencies if your strategy needs them
# dependencies:
#   - "numpy>=1.20"
```
Key changes:
*   `strategy_id`: Changed to \`"awesome_strat"\`, matching our class.
*   `strategy_class_name`: Changed to \`"MyAwesomeStrategy"\`.
*   `display_name`, `version`, `authors`, `description`: Updated with specific details.
### 4. Document Your Strategy and Dependencies
Edit \`MyAwesomeStrategyPackage/README.md\`. Provide clear instructions on what your strategy does, how to use it, any parameters it accepts, and its benefits. A good README is essential for users.
If your strategy has specific Python dependencies (e.g., \`nltk\`, \`scikit-learn\`), add them to \`MyAwesomeStrategyPackage/requirements.txt\`, one per line, like:
```text
# In requirements.txt
numpy>=1.20
scikit-learn==1.2.0
```
### 5. Validate Your Package
Before sharing, use Compact Memory's validation tool to check for common issues:
```bash
compact-memory dev validate-strategy-package path/to/MyAwesomeStrategyPackage
```
This will check the manifest, ensure the strategy module and class can be loaded, and look for a \`requirements.txt\` and \`README.md\`.
### 6. Sharing Your Strategy
Your strategy package directory (\`MyAwesomeStrategyPackage\`) is now ready! Here's how it can be shared and used:
*   `Direct Sharing (Zip/Git):** You can zip the \`MyAwesomeStrategyPackage\` directory and share it. Users can then place it in their Compact Memory plugin directory.`
*   `Python Package (Advanced):** For wider distribution (e.g., via PyPI), you would typically:`
    *   `Add a \`pyproject.toml\` (or \`setup.py\`) to the root of \`MyAwesomeStrategyPackage\` or one level above it.`
    *   `Configure this file to include your strategy files and register your strategy as a plugin using entry points. (Refer to \`docs/SHARING_STRATEGIES.md\` for details on entry points).`
    *   `Build your package (e.g., \`python -m build\`) and upload it to PyPI.`
### 7. Using the Packaged Strategy
Once a user has your package (either by placing the directory in their plugins folder or by \`pip install\`-ing your Python package if you created one), they can use it like any other strategy:
```bash
compact-memory dev list-strategies # Your strategy should appear here
compact-memory compress input.txt --strategy awesome_strat --budget 100 --strategy-params '{"custom_param": 10}'
```
```python
from compact_memory.compression import get_compression_strategy
MyStrategyClass = get_compression_strategy("awesome_strat")
my_strat_instance = MyStrategyClass(custom_param=10)
# ... use my_strat_instance.compress(...) ...
```
## Conclusion
You've successfully taken a strategy idea, used \`dev create-strategy-package\` to structure it, configured its manifest, and prepared it for sharing. This process helps build a robust ecosystem of compression strategies for Compact Memory. For more details on plugin mechanisms and distribution, always refer to \`docs/SHARING_STRATEGIES.md\`.
