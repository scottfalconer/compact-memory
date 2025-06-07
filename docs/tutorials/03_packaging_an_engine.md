# Tutorial 3: From Idea to Plugin - Packaging Your Compact Memory Engine
This tutorial guides you through the process of packaging a custom compression engine you've developed for Compact Memory. Packaging makes your engine easily shareable, discoverable, and installable by others. We'll use the \`dev create-engine-package\` CLI command and prepare the engine for distribution.
## Prerequisites
*   `Compact Memory installed (\`pip install compact-memory\`).`
*   `You have already developed a custom compression engine (e.g., as a \`.py\` file). For this tutorial, we'll assume you have a engine in \`my_awesome_engine.py\` containing a class \`MyAwesomeEngine\` which inherits from \`BaseCompressionEngine\`.`
*   `Your \`MyAwesomeEngine\` class has a unique string attribute \`id = "awesome_strat"\`.`
## Goal
By the end of this tutorial, you will have a well-structured engine package directory ready for sharing, either directly or as part of a larger Python package.
## Steps
### 1. Generate the Package Template
Compact Memory's CLI provides a handy tool to bootstrap a engine package. Open your terminal and navigate to where you want to create your package directory.
```bash
compact-memory dev create-engine-package --name compact_memory_my_engine
```
This command will create a new directory named \`compact_memory_my_engine\` with the following structure:
```text
compact_memory_my_engine/
├── engine.py                      # Template engine implementation
├── engine_package.yaml            # Manifest file
├── README.md                        # Basic README for your package
├── requirements.txt                 # For dependencies
```
### 2. Add Your Engine Code
Now, replace the contents of the template \`compact_memory_my_engine/engine.py\` with your actual engine code from \`my_awesome_engine.py\`.
Let's assume your \`my_awesome_engine.py\` looked something like this:
```python
from compact_memory.engines import BaseCompressionEngine, CompressedMemory, CompressionTrace
# ... any other imports your engine needs ...

class MyAwesomeEngine(BaseCompressionEngine):
    id = "awesome_strat" # Your unique engine ID

    def __init__(self, custom_param: int = 5):
        self.custom_param = custom_param
        # ...

    def compress(self, text_or_chunks, llm_token_budget, **kwargs):
        # Your brilliant compression logic here
        compressed_text = f"Compressed with custom_param {self.custom_param}: {str(text_or_chunks)[:100]}"
        trace = CompressionTrace(
            engine_name=self.id,
            engine_params={'custom_param': self.custom_param},
            input_summary={'original_length': len(str(text_or_chunks))},
            output_summary={'compressed_length': len(compressed_text)},
            final_compressed_object_preview=compressed_text[:50]
        )
        return CompressedMemory(text=compressed_text), trace
```
Copy this class into \`compact_memory_my_engine/engine.py\`. You can delete the template \`MyEngine\` class that was there.
### 3. Update the Manifest File
The \`engine_package.yaml\` file is crucial. It tells Compact Memory about your engine. Open \`compact_memory_my_engine/engine_package.yaml\` and edit it. The template will look like this:
```yaml
package_format_version: "1.0"
engine_id: compact_memory_my_engine # Placeholder from command
engine_class_name: MyEngine         # Placeholder from command
engine_module: engine
display_name: compact_memory_my_engine  # Placeholder from command
version: "0.1.0"
authors: []
description: Describe the engine
```
Modify it to accurately reflect your engine. For our \`MyAwesomeEngine\`, it should become:
```yaml
package_format_version: "1.0"
engine_id: awesome_strat          # Must match the 'id' in your class
engine_class_name: MyAwesomeEngine # Your actual class name
engine_module: engine             # The name of the .py file (without extension)
display_name: "My Awesome Engine"   # A user-friendly name
version: "0.1.0"                      # Your engine's version
authors:
  - "Your Name <your.email@example.com>"
description: "An awesome engine that demonstrates packaging."
# Optional: Add dependencies if your engine needs them
# dependencies:
#   - "numpy>=1.20"
```
Key changes:
*   `engine_id`: Changed to \`"awesome_strat"\`, matching our class.
*   `engine_class_name`: Changed to \`"MyAwesomeEngine"\`.
*   `display_name`, `version`, `authors`, `description`: Updated with specific details.
### 4. Document Your Engine and Dependencies
Edit \`compact_memory_my_engine/README.md\`. Provide clear instructions on what your engine does, how to use it, any parameters it accepts, and its benefits. A good README is essential for users.
If your engine has specific Python dependencies (e.g., \`nltk\`, \`scikit-learn\`), add them to \`compact_memory_my_engine/requirements.txt\`, one per line, like:
```text
# In requirements.txt
numpy>=1.20
scikit-learn==1.2.0
```
### 5. Validate Your Package
Before sharing, use Compact Memory's validation tool to check for common issues:
```bash
compact-memory dev validate-engine-package path/to/compact_memory_my_engine
```
This will check the manifest, ensure the engine module and class can be loaded, and look for a \`requirements.txt\` and \`README.md\`.
### 6. Sharing Your Engine
Your engine package directory (\`compact_memory_my_engine\`) is now ready! Here's how it can be shared and used:
*   `Direct Sharing (Zip/Git):** You can zip the \`compact_memory_my_engine\` directory and share it. Users can then place it in their Compact Memory plugin directory.`
*   `Python Package (Advanced):** For wider distribution (e.g., via PyPI), you would typically:`
    *   `Add a \`pyproject.toml\` (or \`setup.py\`) to the root of \`compact_memory_my_engine\` or one level above it.`
    *   `Configure this file to include your engine files and register your engine as a plugin using entry points. (Refer to \`docs/SHARING_ENGINES.md\` for details on entry points).`
    *   `Build your package (e.g., \`python -m build\`) and upload it to PyPI.`
### 7. Using the Packaged Engine
Once a user has your package (either by placing the directory in their plugins folder or by \`pip install\`-ing your Python package if you created one), they can use it like any other engine:
```bash
compact-memory dev list-engines # Your engine should appear here
compact-memory compress --file input.txt --engine awesome_strat --budget 100 --engine-params '{"custom_param": 10}'
```
```python
from compact_memory import get_compression_engine
MyEngineClass = get_compression_engine("awesome_strat")
my_strat_instance = MyEngineClass(custom_param=10)
# ... use my_strat_instance.compress(...) ...
```
## Conclusion
You've successfully taken a engine idea, used \`dev create-engine-package\` to structure it, configured its manifest, and prepared it for sharing. This process helps build a robust ecosystem of compression engines for Compact Memory. For more details on plugin mechanisms and distribution, always refer to \`docs/SHARING_ENGINES.md\`.
