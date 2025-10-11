# Google Colab Download Examples

## Example 1: Download the entire crossword-solver directory

```python
from download_utils import download_directory

# This will zip and download the entire repository to your local system
download_directory('.')
```

## Example 2: Download specific directories

```python
from download_utils import download_directory

# Download just the LLM solvers
download_directory('llm')

# Download just the digit classifier models
download_directory('two_digit_classifier')
```

## Example 3: Download with custom filename

```python
from download_utils import download_directory

# Save with a custom name
download_directory('.', 'my-custom-name.zip')
```

## Example 4: Download a single file

```python
from download_utils import download_file

# Download the main crossword solver script
download_file('crossword.py')

# Download a specific model file
download_file('train.py')
```

## Running from terminal (in Colab)

You can also run it as a script:

```bash
!python download_utils.py .              # Download everything
!python download_utils.py llm            # Download llm directory
!python download_utils.py crossword.py   # Download single file
```

## Notes

- The utility automatically detects Google Colab environment
- Files are zipped before download to save bandwidth
- Git files (.git directory) are excluded from the zip
- Large model files listed in .gitignore are excluded
- The zip file is created in /tmp/ and then downloaded
