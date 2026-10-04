"""
CricMetrics Pro: Automated Master Documentation & PDF Generator
Builds:
1. docs/MACHINE_LEARNING_GUIDE.md, .html, .pdf
2. docs/PROJECT_ARCHITECTURE_GUIDE.md, .html, .pdf
"""

import os
import subprocess
from build_ml_guide import generate_ml_guide
from build_project_guide import generate_project_guide

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

def compile_pdf(html_rel_path, pdf_rel_path):
    html_abs = os.path.join(DOCS_DIR, html_rel_path)
    pdf_abs = os.path.join(DOCS_DIR, pdf_rel_path)
    print(f"Compiling PDF: {html_rel_path} -> {pdf_rel_path}...")
    cmd = [
        CHROME_PATH,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_abs}",
        html_abs
    ]
    subprocess.run(cmd, check=True)
    size_kb = os.path.getsize(pdf_abs) / 1024
    print(f"Generated: {pdf_rel_path} ({size_kb:.1f} KB)")

if __name__ == "__main__":
    print("Generating Documentation...")
    generate_ml_guide()
    generate_project_guide()
    compile_pdf("MACHINE_LEARNING_GUIDE.html", "MACHINE_LEARNING_GUIDE.pdf")
    compile_pdf("PROJECT_ARCHITECTURE_GUIDE.html", "PROJECT_ARCHITECTURE_GUIDE.pdf")
    print("All documentation and PDFs generated successfully.")
