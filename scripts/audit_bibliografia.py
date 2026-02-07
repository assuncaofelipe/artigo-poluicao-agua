#!/usr/bin/env python3
import os
import re
from collections import defaultdict


BASE = "/home/alfredo/Documentos/artigo_lago"
BIB_PATH = os.path.join(BASE, "artigo", "referencias.bib")
PDF_DIR = os.path.join(BASE, "Bibliografias")
OUT_PATH = os.path.join(BASE, "BIBLIOGRAFIA_AUDIT.txt")


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def extract_entries(bib_text: str):
    entries = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", bib_text, re.S):
        entry_type, key, body = m.group(1), m.group(2), m.group(3)
        title = None
        year = None
        doi = None
        for field, value in re.findall(r"(\w+)\s*=\s*\{([^}]*)\}", body):
            if field.lower() == "title":
                title = value
            elif field.lower() == "year":
                year = value
            elif field.lower() == "doi":
                doi = value
        entries.append(
            {
                "type": entry_type,
                "key": key,
                "title": title,
                "year": year,
                "doi": doi,
            }
        )
    return entries


def main():
    with open(BIB_PATH, encoding="utf-8") as f:
        bib_text = f.read()

    entries = extract_entries(bib_text)
    pdfs = [p for p in os.listdir(PDF_DIR) if p.lower().endswith(".pdf")]

    pdf_norm = {p: normalize(p) for p in pdfs}
    unmatched_pdfs = set(pdfs)

    report_lines = []
    report_lines.append("AUDITORIA DE BIBLIOGRAFIA (BIB vs PDF LOCAL)")
    report_lines.append(f"Total de entradas BibTeX: {len(entries)}")
    report_lines.append(f"Total de PDFs em Bibliografias/: {len(pdfs)}")
    report_lines.append("")

    matches = []
    misses = []

    for e in entries:
        title = e["title"] or ""
        title_norm = normalize(title)
        # match direto por chave (ex.: CONAMA)
        if "conama" in (e["key"] or "").lower():
            for p in pdfs:
                if "conama_357" in p.lower() or "conama" in p.lower():
                    matches.append((e, p, 99))
                    unmatched_pdfs.discard(p)
                    break
            else:
                misses.append(e)
            continue
        best = None
        best_score = 0
        for p, p_norm in pdf_norm.items():
            # score por interseção de palavras
            tokens = set(title_norm.split())
            if not tokens:
                continue
            overlap = len(tokens.intersection(p_norm.split()))
            if overlap > best_score:
                best_score = overlap
                best = p
        if best and best_score >= 3:
            matches.append((e, best, best_score))
            unmatched_pdfs.discard(best)
        else:
            misses.append(e)

    report_lines.append("1) ENTRADAS COM PDF CORRESPONDENTE (match por título)")
    for e, p, score in matches:
        report_lines.append(
            f"- {e['key']} ({e['year']}): \"{e['title']}\" -> {p} (score={score})"
        )

    report_lines.append("")
    report_lines.append("2) ENTRADAS SEM PDF CORRESPONDENTE (verificar manualmente)")
    for e in misses:
        report_lines.append(
            f"- {e['key']} ({e['year']}): \"{e['title']}\" | DOI: {e['doi'] or 'N/A'}"
        )

    report_lines.append("")
    report_lines.append("3) PDFs SEM ENTRADA NO .bib (verificar inclusão)")
    for p in sorted(unmatched_pdfs):
        report_lines.append(f"- {p}")

    report_lines.append("")
    report_lines.append("Observação: a correspondência é baseada em similaridade textual do título.")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Relatório salvo em: {OUT_PATH}")


if __name__ == "__main__":
    main()
