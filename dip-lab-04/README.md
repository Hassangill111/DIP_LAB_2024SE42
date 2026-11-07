# DIP Lab Task 04 - Geometric Transformations with Custom Inputs

**Name:** Syed Muhammad Hassan Gillani  
**Roll No:** 2024-SE-42  **Semester:** 5th (Software Engineering, Session 2024-28)  
**Course:** Digital Image Processing Lab - Lab Instructor: Engr. Fahad Nisar  
**University:** The University of Azad Jammu and Kashmir, Department of Software Engineering

## What this project does
One reusable function `transform(image, operation, **params)` performs **scale, crop, flip, translate and rotate**
with user-supplied values, validates every input, and returns the transformed image.
The notebook demonstrates every operation, compares 4 interpolation methods, shows rejected inputs for each
operation, runs identity checks, and explains coordinates vs indexing, aspect-ratio distortion,
rotation canvas expansion and translation clipping.

## Repository layout
| Path | Content |
|---|---|
| `2024-SET-2024-SE-42-LAB04.ipynb` | Completed, fully executed notebook (outputs visible) |
| `transforms.py` | Helper source: `transform`, validators, affine matrices, prompt loops |
| `make_input_image.py` | Script that generated the input image |
| `input/sample_image.png` | Input image (600 x 400) |
| `outputs/` | Saved transformed images (PNG) |
| `outputs/comparisons/` | Labelled original-vs-result figures |
| `Lab04_Report_2024-SE-42.docx` | Detailed Word report |
| `requirements.txt` | Dependencies |

## Run
```bash
pip install -r requirements.txt
jupyter notebook 2024-SET-2024-SE-42-LAB04.ipynb      # Kernel > Restart & Run All
```
The notebook replays recorded answers so it runs unattended. Set `INTERACTIVE = True` in the setup cell and run
the "live prompt" cell (section 3f) to type your own operation and values.

## Conventions
RGB colour (converted once from OpenCV BGR) - image coordinates (x = column, y = row, origin top-left) -
black background for exposed areas - positive rotation = counter-clockwise -
maximum output 4000 px per side / 16,000,000 px total.
