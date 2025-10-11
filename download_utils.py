"""
Utility script to download directories when working in Google Colab or Jupyter notebooks.

Usage:
    from download_utils import download_directory
    download_directory('.')  # Downloads the entire current directory
    download_directory('llm')  # Downloads the llm directory
"""

import os
import zipfile
from pathlib import Path


def download_directory(directory_path='.', output_filename=None):
    """
    Download a directory by zipping it and triggering a download in Colab/Jupyter.
    
    Args:
        directory_path (str): Path to the directory to download. Default is current directory.
        output_filename (str): Name of the output zip file. If None, uses directory name.
    
    Returns:
        str: Path to the created zip file
    """
    try:
        # Check if running in Colab
        from google.colab import files
        in_colab = True
    except ImportError:
        in_colab = False
        print("Not running in Google Colab. Will create zip file locally.")
    
    # Convert to Path object for easier handling
    dir_path = Path(directory_path).resolve()
    
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory '{directory_path}' does not exist")
    
    # Determine output filename
    if output_filename is None:
        if dir_path.name == '.' or dir_path == Path.cwd():
            output_filename = 'crossword-solver.zip'
        else:
            output_filename = f'{dir_path.name}.zip'
    
    if not output_filename.endswith('.zip'):
        output_filename += '.zip'
    
    zip_path = Path('/tmp') / output_filename
    
    # Create zip file
    print(f"Creating zip file: {zip_path}")
    
    # Patterns to exclude
    exclude_patterns = ['.git', '__pycache__', '.pyc', '.pyo', '.DS_Store', '.pytest_cache']
    exclude_files = ['.gitignore', '.env', '.env.local']
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        if dir_path.is_file():
            zipf.write(dir_path, dir_path.name)
            print(f"Added file: {dir_path.name}")
        else:
            # Walk through directory and add files
            for root, dirs, files in os.walk(dir_path):
                # Skip excluded directories
                dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]
                
                # Skip if current directory matches exclude patterns
                if any(pattern in root for pattern in exclude_patterns):
                    continue
                
                for file in files:
                    # Skip excluded file patterns and specific files
                    if (any(file.endswith(pattern) or pattern in file for pattern in exclude_patterns) or
                        file in exclude_files):
                        continue
                        
                    file_path = Path(root) / file
                    # Get relative path for archive
                    arcname = file_path.relative_to(dir_path.parent)
                    zipf.write(file_path, arcname)
                    print(f"Added: {arcname}")
    
    print(f"\nZip file created successfully: {zip_path}")
    print(f"Zip file size: {zip_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Download in Colab
    if in_colab:
        print("\nTriggering download in Colab...")
        files.download(str(zip_path))
        print("Download started!")
    else:
        print(f"\nZip file saved at: {zip_path}")
        print("To download in Colab, run this script in a Colab environment.")
    
    return str(zip_path)


def download_file(file_path):
    """
    Download a single file in Colab/Jupyter.
    
    Args:
        file_path (str): Path to the file to download
    
    Returns:
        str: Path to the file
    """
    try:
        from google.colab import files
        in_colab = True
    except ImportError:
        in_colab = False
        print("Not running in Google Colab.")
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File '{file_path}' does not exist")
    
    if not file_path.is_file():
        raise ValueError(f"'{file_path}' is not a file. Use download_directory() for directories.")
    
    if in_colab:
        print(f"Downloading file: {file_path}")
        files.download(str(file_path))
        print("Download started!")
    else:
        print(f"File location: {file_path.resolve()}")
        print("To download in Colab, run this script in a Colab environment.")
    
    return str(file_path)


if __name__ == "__main__":
    # Example usage
    print("Download Utilities for Google Colab")
    print("=" * 50)
    print("\nExample 1: Download entire directory")
    print("  download_directory('.')")
    print("\nExample 2: Download specific directory")
    print("  download_directory('llm')")
    print("\nExample 3: Download with custom filename")
    print("  download_directory('.', 'my-crossword-solver.zip')")
    print("\nExample 4: Download a single file")
    print("  download_file('crossword.py')")
    print("\n" + "=" * 50)
    
    # Interactive mode
    import sys
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if Path(target).is_file():
            download_file(target)
        else:
            download_directory(target)
    else:
        print("\nUsage: python download_utils.py <directory_or_file>")
        print("Or import in your notebook: from download_utils import download_directory")
