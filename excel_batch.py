"""
excel_batch.py
--------------
Processa uma planilha Excel em lote: lê uma coluna de textos, analisa o
sentimento de cada linha e gera uma nova planilha com os resultados.

Uso:
    python excel_batch.py entrada.xlsx
    python excel_batch.py entrada.xlsx --coluna "Comentario" --saida resultado.xlsx

A planilha de entrada precisa ter pelo menos uma coluna com os textos a
analisar (por padrão, o script procura uma coluna chamada "texto",
"comentario", "comentário", "review" ou "mensagem" - case-insensitive -
ou você pode indicar explicitamente com --coluna).
"""

import argparse
import sys
import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

from sentiment_core import analisar_lote

COLUNAS_CANDIDATAS = ["texto", "comentario", "comentário", "review", "mensagem", "feedback"]

CORES_PREENCHIMENTO = {
    "Positivo": "C6EFCE",  # verde claro
    "Negativo": "FFC7CE",  # vermelho claro
    "Neutro": "F2F2F2",    # cinza claro
}
CORES_FONTE = {
    "Positivo": "006100",
    "Negativo": "9C0006",
    "Neutro": "606060",
}


def detectar_coluna(df: pd.DataFrame) -> str:
    """Tenta descobrir automaticamente qual coluna contém os textos."""
    colunas_lower = {c.lower().strip(): c for c in df.columns}
    for candidata in COLUNAS_CANDIDATAS:
        if candidata in colunas_lower:
            return colunas_lower[candidata]
    raise ValueError(
        f"Não encontrei automaticamente a coluna de texto. "
        f"Colunas disponíveis: {list(df.columns)}. "
        f"Use --coluna \"NomeDaColuna\" para indicar manualmente."
    )


def processar_planilha(caminho_entrada: str, coluna: str | None, caminho_saida: str) -> str:
    df = pd.read_excel(caminho_entrada)

    if df.empty:
        raise ValueError("A planilha de entrada está vazia.")

    if coluna is None:
        coluna = detectar_coluna(df)
    elif coluna not in df.columns:
        raise ValueError(f"Coluna '{coluna}' não encontrada. Colunas disponíveis: {list(df.columns)}")

    print(f"Coluna de texto identificada: '{coluna}'")
    print(f"Total de linhas a processar: {len(df)}")

    textos = df[coluna].fillna("").astype(str).tolist()
    resultados = analisar_lote(textos)

    # Monta as novas colunas a partir dos resultados
    df_resultado = df.copy()
    df_resultado["Sentimento"] = [r["classificacao"] for r in resultados]
    df_resultado["Score (compound)"] = [r["compound"] for r in resultados]
    df_resultado["Negativo (%)"] = [r["neg"] for r in resultados]
    df_resultado["Neutro (%)"] = [r["neu"] for r in resultados]
    df_resultado["Positivo (%)"] = [r["pos"] for r in resultados]
    df_resultado["Texto traduzido (EN)"] = [r["texto_traduzido"] for r in resultados]

    df_resultado.to_excel(caminho_saida, index=False, sheet_name="Análise de Sentimento")

    _formatar_planilha(caminho_saida, df_resultado)

    n_erros = sum(1 for r in resultados if r["classificacao"].startswith("ERRO"))
    if n_erros:
        print(f"⚠️  {n_erros} linha(s) tiveram erro de tradução/análise (ver coluna 'Sentimento').")

    return caminho_saida


def _formatar_planilha(caminho: str, df: pd.DataFrame):
    """Aplica formatação profissional: fonte, cabeçalho, cores por sentimento, largura de coluna."""
    from openpyxl import load_workbook

    wb = load_workbook(caminho)
    ws = wb.active

    fonte_padrao = "Arial"
    idx_sentimento = list(df.columns).index("Sentimento") + 1  # 1-indexed

    # Cabeçalho
    for cell in ws[1]:
        cell.font = Font(name=fonte_padrao, bold=True, color="FFFFFF")
        cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        cell.alignment = Alignment(horizontal="center", vertical="center")

    # Linhas de dados
    for row_idx in range(2, ws.max_row + 1):
        sentimento = ws.cell(row=row_idx, column=idx_sentimento).value
        cor_fundo = CORES_PREENCHIMENTO.get(sentimento, "FFFFFF")
        cor_fonte = CORES_FONTE.get(sentimento, "000000")

        for col_idx in range(1, ws.max_column + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.font = Font(name=fonte_padrao, color=cor_fonte if col_idx == idx_sentimento else "000000")
            if col_idx == idx_sentimento:
                cell.fill = PatternFill(start_color=cor_fundo, end_color=cor_fundo, fill_type="solid")

    # Largura automática (aproximada) das colunas
    for col_idx, col_name in enumerate(df.columns, start=1):
        letra = get_column_letter(col_idx)
        maior_valor = max(
            [len(str(col_name))] + [len(str(v)) for v in df[col_name].astype(str).tolist()[:200]]
        )
        ws.column_dimensions[letra].width = min(max(maior_valor + 2, 12), 50)

    ws.freeze_panes = "A2"  # congela o cabeçalho
    wb.save(caminho)


def main():
    parser = argparse.ArgumentParser(description="Analisa sentimento de uma planilha Excel em lote.")
    parser.add_argument("entrada", help="Caminho da planilha .xlsx de entrada")
    parser.add_argument("--coluna", default=None, help="Nome da coluna com os textos (opcional, autodetecta)")
    parser.add_argument("--saida", default=None, help="Caminho da planilha de saída (opcional)")
    args = parser.parse_args()

    caminho_saida = args.saida or args.entrada.replace(".xlsx", "_analisado.xlsx")

    try:
        resultado_path = processar_planilha(args.entrada, args.coluna, caminho_saida)
        print(f"\n✅ Concluído! Planilha gerada em: {resultado_path}")
    except Exception as e:
        print(f"\n❌ Erro: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()