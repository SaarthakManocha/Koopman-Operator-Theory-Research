#!/bin/bash
# Build the Word version of the paper: run after the PDF build (needs main.aux and main.bbl).
set -e
cd "$(dirname "$0")/.."
python3 word/prep_docx.py
cd word
pandoc main_docx.tex -f latex -t docx --reference-doc=ref.docx --resource-path=..:../figs -o Koopman_Linux_Scheduler_Paper.docx
python3 post_docx.py Koopman_Linux_Scheduler_Paper.docx
