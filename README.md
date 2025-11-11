# crossword-solver

OpenCV + Tesseract + LLM-based crossword puzzle solver.

## Overview
This project converts a photographed crossword into a structured JSON puzzle, queries LLM solvers for answers, and validates/visualizes the results. The pipeline is modular: image processing → JSON construction → LLM integration → solution validation.

## Key features
- Grid extraction and per-cell OCR (Tesseract).
- JSON puzzle builder with word lengths and intersections.
- Pluggable LLM solver integrations (multiple providers supported).

## Requirements
- Python 3.8+
- OpenCV (cv2), Pillow, pytesseract
- API keys for any LLM providers used (configured in project)

## Quick start
1. Install dependencies (example):
   pip install -r requirements.txt
2. Ensure Tesseract is installed and pytesseract.tesseract_cmd points to the binary.
3. Run the main program:
   python crossword.py 

## Project layout (high level)
- crossword.py — main pipeline and orchestration  
- digit_recogniser.py — cell extraction and OCR helpers  
- digit_classifiers/ — per-detector implementations (Tesseract-based, etc.)  
- generate_json.py — builds puzzle JSON (lengths, intersections)  
- llm/ — LLM solver wrappers and prompts  
- overlay_grid.py — visualization/overlay of solutions
