$ErrorActionPreference = "Stop"

python download_osm.py

foreach ($seed in 11, 22, 33, 44, 55) {
    python run_paper_benchmark.py `
        --seed $seed `
        --synthetic-per-mode 800 `
        --osm-per-site-mode 100 `
        --epochs 80 `
        --output "outputs/paper/seed_$seed"
}

python analyze_paper_results.py --input outputs/paper --output outputs/paper_summary
python run_scaling_benchmark.py

foreach ($seed in 11, 22, 33, 44, 55) {
    python run_geographic_transfer.py `
        --seed $seed `
        --scenarios-per-site-mode 40 `
        --epochs 25 `
        --output "outputs/geographic_transfer/seed_$seed"
}

python analyze_journal_results.py
python collect_environment.py
python create_method_overview.py

Push-Location report
try {
    New-Item -ItemType Directory -Force output/pdf | Out-Null
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf main.tex
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf main.tex
    Copy-Item output/pdf/main.pdf output/pdf/ResiliRoad_Final_Paper.pdf -Force
    Copy-Item output/pdf/main.pdf output/pdf/ResiliRoad_Preliminary_Report.pdf -Force
}
finally {
    Pop-Location
}

Push-Location journal
try {
    New-Item -ItemType Directory -Force output/pdf | Out-Null
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf main.tex
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf main.tex
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf cover_letter.tex
    Copy-Item output/pdf/main.pdf output/pdf/ResiliRoad_JCN_Manuscript.pdf -Force
    Copy-Item output/pdf/cover_letter.pdf output/pdf/ResiliRoad_JCN_Cover_Letter.pdf -Force
}
finally {
    Pop-Location
}

Push-Location poster
try {
    New-Item -ItemType Directory -Force output/pdf | Out-Null
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf main.tex
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf main.tex
    Copy-Item output/pdf/main.pdf output/pdf/ResiliRoad_VMS60_Poster.pdf -Force
}
finally {
    Pop-Location
}
