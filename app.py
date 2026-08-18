import io
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from sentiment_core import analisar_texto
from excel_batch import processar_dataframe, gerar_planilha_formatada, detectar_coluna

# ---------- Configuração da página ----------
st.set_page_config(
    page_title="Analisador de Sentimentos VADER (PT-BR)",
    page_icon="🧠",
    layout="centered"
)

CORES = {"Positivo": "#22c55e", "Negativo": "#ef4444", "Neutro": "#9aa79a"}
EMOJIS = {"Positivo": "😊", "Negativo": "😠", "Neutro": "😐"}

# ---------- Interface ----------
st.title("🧠 Analisador de Sentimentos VADER (PT-BR)")
st.caption(
    "Combina o léxico **VADER** (NLTK) com tradução automática via "
    "**deep_translator**, permitindo analisar sentimentos em textos "
    "escritos em português."
)

with st.expander("ℹ️ Como funciona?"):
    st.markdown(
        """
        1. Você digita uma frase (ou envia uma planilha) em português.
        2. O texto é traduzido automaticamente para inglês.
        3. O VADER analisa o sentimento do texto traduzido.
        4. O resultado (positivo, negativo ou neutro) é exibido com um gráfico.

        **Stack:** Python · NLTK (VADER) · deep_translator · Matplotlib · Pandas

        **Nota sobre limitações:** por ser baseado em léxico (dicionário de
        palavras), o modelo pode ocasionalmente interpretar mal frases
        puramente descritivas com palavras de conotação leve (ex: "original",
        "novo"). A classificação usa regras ajustadas para reduzir esse tipo
        de falso positivo/negativo.
        """
    )

modo = st.radio(
    "O que você quer analisar?",
    ["Uma frase", "Planilha (lote)"],
    horizontal=True,
)

# =============================================================================
# MODO 1: uma frase por vez
# =============================================================================
if modo == "Uma frase":
    texto = st.text_area(
        "Digite uma frase para análise:",
        placeholder="Ex: Eu amo muito esse produto, é simplesmente incrível e maravilhoso!",
        height=100
    )

    analisar = st.button("🔍 Analisar sentimento", type="primary", use_container_width=True)

    if analisar:
        if not texto.strip():
            st.error("Digite algum texto antes de analisar.")
        else:
            with st.spinner("Traduzindo e analisando..."):
                resultado = analisar_texto(texto)

            classificacao = resultado["classificacao"]
            cor = CORES[classificacao]
            emoji = EMOJIS[classificacao]
            polaridade = resultado["compound"]

            st.divider()
            col1, col2, col3 = st.columns(3)
            col1.metric("Resultado", f"{classificacao} {emoji}")
            col2.metric("Score (compound)", f"{polaridade:.2f}")
            col3.metric("Confiança", f"{max(resultado['neg'], resultado['neu'], resultado['pos']):.0%}")

            st.caption(f"**Texto traduzido (uso interno):** _{resultado['texto_traduzido']}_")

            # ---------- Gráfico ----------
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

            ax1.bar(['Sentimento'], [polaridade], color=cor, width=0.4)
            ax1.axhline(0, color='black', linewidth=1)
            ax1.set_ylim(-1.1, 1.1)
            ax1.set_ylabel("Score (-1 a +1)")
            ax1.set_title(f"Sentimento Geral: {classificacao} {emoji}")
            ax1.grid(axis='y', linestyle='--', alpha=0.5)

            va = 'bottom' if polaridade >= 0 else 'top'
            offset = 0.05 if polaridade >= 0 else -0.05
            ax1.text(0, polaridade + offset, f'{polaridade:.2f}', ha='center', va=va, fontweight='bold')

            categorias = ['Negativo', 'Neutro', 'Positivo']
            valores = [resultado['neg'], resultado['neu'], resultado['pos']]
            cores_barras = ['#ef4444', '#9aa79a', '#22c55e']

            ax2.bar(categorias, valores, color=cores_barras, width=0.5)
            ax2.set_ylim(0, 1.1)
            ax2.set_ylabel("Proporção (soma = 1.0)")
            ax2.set_title("Composição do Sentimento")
            ax2.grid(axis='y', linestyle='--', alpha=0.5)
            for i, v in enumerate(valores):
                ax2.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')

            fig.suptitle(f'Frase analisada: "{texto}"', fontsize=10)
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            st.pyplot(fig)

            st.divider()

# =============================================================================
# MODO 2: planilha em lote
# =============================================================================
else:
    st.markdown(
        "Envie uma planilha **.xlsx** ou **.csv** com uma coluna de textos "
        "(ex: comentários de clientes, avaliações, respostas de pesquisa). "
        "O sistema detecta automaticamente a coluna se ela se chamar "
        "*texto*, *comentario*, *review*, *mensagem* ou *feedback* — "
        "ou você pode escolher manualmente depois do upload."
    )

    arquivo = st.file_uploader("Planilha de entrada", type=["xlsx", "csv"])

    if arquivo is not None:
        try:
            if arquivo.name.lower().endswith(".csv"):
                df_entrada = pd.read_csv(arquivo, sep=None, engine="python")
            else:
                df_entrada = pd.read_excel(arquivo)
        except Exception as e:
            st.error(f"Não consegui ler o arquivo enviado: {e}")
            df_entrada = None

        if df_entrada is not None:
            st.write(f"**{len(df_entrada)} linha(s)** encontradas. Pré-visualização:")
            st.dataframe(df_entrada.head(5), use_container_width=True)

            try:
                coluna_sugerida = detectar_coluna(df_entrada)
            except ValueError:
                coluna_sugerida = df_entrada.columns[0]

            coluna_escolhida = st.selectbox(
                "Coluna que contém os textos a analisar:",
                options=list(df_entrada.columns),
                index=list(df_entrada.columns).index(coluna_sugerida),
            )

            processar = st.button("🔍 Analisar planilha", type="primary", use_container_width=True)

            if processar:
                barra = st.progress(0, text="Iniciando análise em lote...")
                with st.spinner(f"Analisando {len(df_entrada)} linha(s)... isso pode levar um tempo."):
                    df_resultado = processar_dataframe(df_entrada, coluna_escolhida)
                barra.progress(100, text="Concluído!")

                st.divider()
                st.success(f"Análise concluída para {len(df_resultado)} linha(s).")

                # Resumo rápido
                contagem = df_resultado["Sentimento"].value_counts()
                col1, col2, col3 = st.columns(3)
                col1.metric("😊 Positivo", int(contagem.get("Positivo", 0)))
                col2.metric("😠 Negativo", int(contagem.get("Negativo", 0)))
                col3.metric("😐 Neutro", int(contagem.get("Neutro", 0)))

                st.dataframe(
                    df_resultado[[coluna_escolhida, "Sentimento", "Score (compound)"]],
                    use_container_width=True,
                )

                # Gera o .xlsx formatado em memória para download (sem gravar em disco)
                buffer = io.BytesIO()
                gerar_planilha_formatada(df_resultado, buffer)

                st.download_button(
                    label="⬇️ Baixar planilha analisada (.xlsx)",
                    data=buffer,
                    file_name="planilha_analisada.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

st.divider()
st.caption("Desenvolvido por Andressa Gabriel · [GitHub](https://github.com/Andressagabriell)")