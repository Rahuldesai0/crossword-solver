# crossword-solver
OpenCV Python Crossword puzzle solver using LLMs 

## Download Utility for Google Colab

When working in Google Colab, you can easily download the entire repository or specific directories to your local system using the included download utility.

### Usage

```python
from download_utils import download_directory, download_file

# Download the entire repository
download_directory('.')

# Download a specific directory
download_directory('llm')

# Download with a custom filename
download_directory('.', 'my-crossword-solver.zip')

# Download a single file
download_file('crossword.py')
```

### Command Line Usage

You can also use it from the command line:

```bash
python download_utils.py .              # Download entire directory
python download_utils.py llm            # Download llm directory
python download_utils.py crossword.py   # Download single file
```

The utility will automatically detect if you're running in Google Colab and trigger a download. If not in Colab, it will create a zip file in `/tmp/`.
