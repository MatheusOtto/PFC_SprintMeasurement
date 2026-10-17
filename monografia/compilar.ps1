# Compila a monografia: pdflatex -> bibtex -> pdflatex -> pdflatex (sem Perl nem latexmk).
# Uso (PowerShell, de qualquer pasta):  .\monografia\compilar.ps1
# Resultado: monografia\main.pdf (não versionado). Termina com código 1 se o log tiver erro.

$miktex = Join-Path $env:LOCALAPPDATA "Programs\MiKTeX\miktex\bin\x64"
if (Test-Path $miktex) { $env:PATH = "$miktex;$env:PATH" }

Set-Location $PSScriptRoot

# cmd /c evita que o aviso do MiKTeX no stderr vire erro do PowerShell.
$passos = @(
    "pdflatex --enable-installer -interaction=nonstopmode -file-line-error main.tex",
    "bibtex main",
    "pdflatex --enable-installer -interaction=nonstopmode -file-line-error main.tex",
    "pdflatex --enable-installer -interaction=nonstopmode -file-line-error main.tex"
)
foreach ($p in $passos) { cmd /c "$p >nul 2>&1" }

$log = Get-Content main.log -Encoding UTF8
$erros = $log | Select-String -Pattern '^(\S+\.tex:\d+:|!)' | ForEach-Object { $_.Line.Trim() }
$pendentes = $log | Select-String -Pattern "(Citation|Reference) .* undefined" | ForEach-Object { $_.Line.Trim() }
$paginas = ($log | Select-String -Pattern 'Output written on main.pdf \((\d+) page' |
            ForEach-Object { $_.Matches[0].Groups[1].Value } | Select-Object -Last 1)

"Paginas: $paginas"
"Erros de LaTeX: $(@($erros).Count)"
$erros | Select-Object -First 15 | ForEach-Object { "  $_" }
"Citacoes/referencias indefinidas: $(@($pendentes).Count)"
$pendentes | Select-Object -First 15 | ForEach-Object { "  $_" }

if (@($erros).Count -gt 0 -or @($pendentes).Count -gt 0) { exit 1 }
