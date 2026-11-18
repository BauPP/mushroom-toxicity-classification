# -*- coding: utf-8 -*-
"""Exporta un documento Markdown de docs/ a PDF con ecuaciones renderizadas
(KaTeX auto-render) y las figuras embebidas.
Salida: docs/<nombre>.pdf

Uso: python exportar_formulas_pdf.py [nombre] [--ieee]
     nombre por defecto: formulas_matematicas_modelos  (p. ej. informe_final)
     --ieee: formato de artículo IEEE (dos columnas, Times, ecuaciones numeradas)
     Requiere Microsoft Edge en la ruta estándar.
"""
import base64
import mimetypes
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent
IEEE = "--ieee" in sys.argv
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
NOMBRE = _args[0] if _args else "formulas_matematicas_modelos"
MD_PATH = ROOT / "docs" / f"{NOMBRE}.md"
PDF_PATH = ROOT / "docs" / f"{NOMBRE}.pdf"
EDGE_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]

texto = MD_PATH.read_text(encoding="utf-8")

# Proteger las ecuaciones de la conversión markdown (los _ y ^ se perderían)
formulas = []


def _guardar(m):
    formulas.append(m.group(0))
    return f"MATHPLACEHOLDER{len(formulas) - 1}ENDMATH"


patron = re.compile(r"\$\$.*?\$\$|\$[^$\n]+\$", re.DOTALL)
texto_protegido = patron.sub(_guardar, texto)

html_cuerpo = markdown.markdown(
    texto_protegido,
    extensions=["tables", "fenced_code", "sane_lists", "attr_list", "md_in_html"],
)
num_ecuacion = 0
for i, f in enumerate(formulas):
    if IEEE and f.startswith("$$"):
        # Formato de ecuación IEEE: centrada y numerada consecutivamente a la derecha.
        num_ecuacion += 1
        f = f"$${f[2:-2].strip()}\\tag{{{num_ecuacion}}}$$"
    # KaTeX auto-render procesa los pares $...$ / $$...$$ de forma determinista.
    html_cuerpo = html_cuerpo.replace(f"MATHPLACEHOLDER{i}ENDMATH", f)

if IEEE:
    # El título y los autores ocupan todo el ancho; el resto fluye en dos columnas.
    cabecera, _, resto = html_cuerpo.partition("<hr />")
    html_cuerpo = f'<header class="titulo">{cabecera}</header>\n<main class="cols">{resto}</main>'
    print(f"Ecuaciones numeradas: {num_ecuacion}")


# El HTML se imprime desde una carpeta temporal, así que las rutas relativas de las
# imágenes no resolverían: se incrustan como data URI (base64).
def _incrustar_imagen(m):
    ruta = (MD_PATH.parent / m.group(2)).resolve()
    if not ruta.exists():
        raise SystemExit(f"Imagen no encontrada: {ruta}")
    mime = mimetypes.guess_type(ruta.name)[0] or "image/png"
    datos = base64.b64encode(ruta.read_bytes()).decode("ascii")
    return f'{m.group(1)}data:{mime};base64,{datos}{m.group(3)}'


html_cuerpo = re.sub(r'(<img[^>]*?src=")(?!data:|https?:)([^"]+)(")', _incrustar_imagen, html_cuerpo)

plantilla = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Fórmulas Matemáticas de los Modelos de Machine Learning</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {delimiters: [{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}], throwOnError:false});"></script>
<style>
  body { font-family: "Segoe UI", Arial, sans-serif; margin: 0; color: #1a1a1a;
         font-size: 13px; line-height: 1.55; overflow-wrap: break-word; }
  h1 { color: #14385c; font-size: 24px; border-bottom: 3px solid #14385c; padding-bottom: 8px; }
  h2 { color: #1f5382; font-size: 19px; border-bottom: 1px solid #c9d7e4; padding-bottom: 5px; margin-top: 28px; }
  h3 { color: #2a6496; font-size: 15.5px; margin-top: 20px; }
  table { border-collapse: collapse; margin: 14px 0; font-size: 11.5px; width: 100%; page-break-inside: avoid; }
  th, td { border: 1px solid #b9c6d2; padding: 5px 7px; text-align: left; }
  .katex-display { overflow: visible !important; }
  ::-webkit-scrollbar { display: none; }
  img { page-break-inside: avoid; }
  p:has(> img) + p > em { display: block; text-align: center; font-size: 12px; color: #444; }
  th { background: #eef3f8; }
  tr:nth-child(even) td { background: #f7fafc; }
  code { background: #f0f2f5; padding: 1px 5px; border-radius: 3px; font-size: 12px; }
  img { max-width: 100%; height: auto; display: block; margin: 12px auto; border: 1px solid #dfe6ec;
        border-radius: 4px; padding: 4px; }
  .katex-display { margin: 14px 0 !important; }
  blockquote { border-left: 4px solid #2a6496; margin-left: 0; padding-left: 14px; color: #444; }
  hr { border: none; border-top: 1px solid #c9d7e4; margin: 22px 0; }
  @page { size: A4; margin: 16mm 15mm; }
</style>
</head>
<body>
__CUERPO__
</body>
</html>
"""

ESTILO_IEEE = """<style>
  @page { size: A4; margin: 19mm 14mm 22mm 14mm; }
  body { font-family: "Times New Roman", Times, serif; font-size: 10pt; line-height: 1.16;
         color: #000; margin: 0; text-align: justify; hyphens: auto; overflow-wrap: break-word; }
  header.titulo { text-align: center; margin-bottom: 7mm; }
  header.titulo h1 { font-size: 22pt; font-weight: normal; line-height: 1.15; color: #000;
         border: none; margin: 0 6mm 5mm; padding: 0; }
  .autores { font-size: 10.5pt; line-height: 1.3; margin: 0; text-indent: 0; text-align: center; }
  main.cols { column-count: 2; column-gap: 6mm; }
  p { margin: 0; text-indent: 1em; }
  .abstract { font-weight: bold; font-size: 9pt; margin-bottom: 2.5mm; }
  h2 { font-size: 10pt; font-weight: normal; font-variant: small-caps; text-align: center; color: #000;
       border: none; padding: 0; margin: 4.5mm 0 2mm; break-after: avoid; letter-spacing: .3px; }
  h3 { font-size: 10pt; font-weight: normal; font-style: italic; text-align: left; color: #000;
       margin: 3mm 0 1.2mm; break-after: avoid; }
  ul, ol { margin: 1mm 0; padding-left: 1.5em; }
  li p { text-indent: 0; }
  table { border-collapse: collapse; width: 100%; font-size: 8pt; margin: 0 0 3mm; text-align: left;
          break-inside: avoid; hyphens: manual; }
  th, td { border: none; padding: 1.5px 3px; background: none !important; vertical-align: top; }
  thead th { border-top: 1px solid #000; border-bottom: 0.6px solid #000; font-weight: bold; }
  tbody tr:last-child td { border-bottom: 1px solid #000; }
  .tcap { text-align: center; text-indent: 0; font-size: 8pt; font-variant: small-caps;
          margin: 3mm 0 1.2mm; line-height: 1.3; break-after: avoid; }
  p:has(> img) { text-indent: 0; margin: 3mm 0 0; break-inside: avoid; break-after: avoid; }
  img { width: 100%; height: auto; display: block; border: none; padding: 0; margin: 0; }
  .fcap { font-size: 8pt; text-indent: 0; text-align: justify; margin: 1.2mm 0 3mm; }
  code { font-family: Consolas, "Courier New", monospace; font-size: 7.8pt; background: none; padding: 0;
         overflow-wrap: anywhere; }
  .katex { font-size: 1em; }
  .katex-display { margin: 2.2mm 0 !important; overflow: visible !important; }
  .refs p { font-size: 8pt; text-indent: -1.9em; padding-left: 1.9em; margin-bottom: .6mm; text-align: justify;
            overflow-wrap: anywhere; }
  hr { display: none; }
  ::-webkit-scrollbar { display: none; }
</style>"""

if IEEE:
    plantilla = re.sub(r"<style>.*?</style>", lambda _: ESTILO_IEEE, plantilla, flags=re.DOTALL)

html_final = plantilla.replace("__CUERPO__", html_cuerpo)

with tempfile.TemporaryDirectory(dir=ROOT / "docs") as tmp:
    html_path = Path(tmp) / "formulas.html"
    html_path.write_text(html_final, encoding="utf-8")

    edge = next((e for e in EDGE_CANDIDATES if e.exists()), None)
    if edge is None:
        raise SystemExit("No se encontró Microsoft Edge para imprimir a PDF.")

    resultado = subprocess.run(
        [
            str(edge),
            "--headless",
            "--disable-gpu",
            "--no-first-run",
            "--no-pdf-header-footer",
            "--virtual-time-budget=30000",
            f"--print-to-pdf={PDF_PATH}",
            html_path.resolve().as_uri(),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    print("edge exit:", resultado.returncode)

tamaño = PDF_PATH.stat().st_size / 1024
print(f"OK - PDF generado: {PDF_PATH} ({tamaño:.0f} KB)")
