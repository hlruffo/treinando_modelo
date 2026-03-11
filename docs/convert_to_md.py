#!/usr/bin/env python3
"""
Converte arquivos DOC/DOCX ou PDF para Markdown usando o pacote docling.

Uso:
    python convert_to_md.py arquivo.pdf
    python convert_to_md.py arquivo.docx
    python convert_to_md.py pasta/com/documentos/
    python convert_to_md.py arquivo.pdf -o saida.md
    python convert_to_md.py pasta/ -o pasta_saida/
"""

import argparse
import sys
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx"}


def convert_file(input_path: Path, output_path: Path) -> bool:
    """Converte um único arquivo para Markdown. Retorna True em caso de sucesso."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        print("Erro: pacote 'docling' não encontrado. Instale com: pip install docling")
        sys.exit(1)

    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(f"  [IGNORADO] {input_path.name} — extensão não suportada")
        return False

    print(f"  Convertendo: {input_path.name} ...", end=" ", flush=True)
    try:
        converter = DocumentConverter()
        result = converter.convert(str(input_path))
        markdown = result.document.export_to_markdown()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
        print(f"OK → {output_path}")
        return True
    except Exception as e:
        print(f"ERRO\n  {e}")
        return False


def collect_files(source: Path) -> list[Path]:
    """Retorna lista de arquivos suportados a partir de um arquivo ou diretório."""
    if source.is_file():
        return [source]
    return sorted(
        f for f in source.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def resolve_output(input_path: Path, output_arg: str | None, base_input: Path) -> Path:
    """Calcula o caminho de saída .md para um arquivo de entrada."""
    if output_arg is None:
        return input_path.with_suffix(".md")

    output = Path(output_arg)
    if base_input.is_dir():
        # Preserva estrutura relativa dentro do diretório de saída
        relative = input_path.relative_to(base_input)
        return (output / relative).with_suffix(".md")

    # Saída explícita para arquivo único
    if output.is_dir():
        return output / input_path.with_suffix(".md").name
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Converte PDF/DOC/DOCX para Markdown via docling."
    )
    parser.add_argument(
        "input",
        help="Arquivo (.pdf/.doc/.docx) ou diretório com documentos.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Arquivo .md de saída (para entrada única) ou diretório de saída "
            "(para entrada de diretório). Padrão: mesmo local e nome do arquivo original."
        ),
        default=None,
    )
    args = parser.parse_args()

    source = Path(args.input)
    if not source.exists():
        print(f"Erro: '{source}' não encontrado.")
        sys.exit(1)

    files = collect_files(source)
    if not files:
        print(f"Nenhum arquivo suportado encontrado em '{source}'.")
        sys.exit(1)

    total = len(files)
    print(f"\nDocumentos encontrados: {total}\n")

    ok = 0
    for file in files:
        out = resolve_output(file, args.output, source)
        if convert_file(file, out):
            ok += 1

    print(f"\nConcluído: {ok}/{total} arquivo(s) convertido(s).")
    if ok < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
