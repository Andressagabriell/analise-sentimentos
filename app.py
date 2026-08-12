import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
import matplotlib.pyplot as plt

# ---------- Configuração da página ----------
st.set_page_config(
    page_title="Analisador de Sentimentos VADER (PT-BR)",
    page_icon="🧠",
    layout="centered"
)

# ---------- Garante que o léxico do VADER esteja baixado ----------
@st.cache_resource
def carregar_lexico():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')
    return SentimentIntensityAnalyzer()

analyzer = carregar_lexico()


def traduzir_para_ingles(texto):
    """
    O VADER só entende inglês. Como o usuário digita em português,
    traduzimos o texto antes de analisar.
    """
    try:
        return GoogleTranslator(source='pt', target='en').translate(texto)
    except Exception as e:
        st.warning(f"Aviso: falha na tradução ({e}). Analisando o texto original.")
        return texto


def analisar_sentimento_vader(texto_em_ingles):
    return analyzer.polarity_scores(texto_em_ingles)


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
        1. Você digita uma frase em português.
        2. O texto é traduzido automaticamente para inglês.
        3. O VADER analisa o sentimento do texto traduzido.
        4. O resultado (positivo, negativo ou neutro) é exibido com um gráfico.

        **Stack:** Python · NLTK (VADER) · deep_translator · Matplotlib
        """
    )

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
            texto_traduzido = traduzir_para_ingles(texto)
            scores = analisar_sentimento_vader(texto_traduzido)
            polaridade = scores['compound']

        if polaridade >= 0.05:
            resultado, cor, emoji = "Positivo", '#22c55e', "😊"
        elif polaridade <= -0.05:
            resultado, cor, emoji = "Negativo", '#ef4444', "😠"
        else:
            resultado, cor, emoji = "Neutro", '#9aa79a', "😐"

        st.divider()

        col1, col2, col3 = st.columns(3)
        col1.metric("Resultado", f"{resultado} {emoji}")
        col2.metric("Score (compound)", f"{polaridade:.2f}")
        col3.metric("Confiança", f"{max(scores['neg'], scores['neu'], scores['pos']):.0%}")

        st.caption(f"**Texto traduzido (uso interno):** _{texto_traduzido}_")

        # ---------- Gráfico ----------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.bar(['Sentimento'], [polaridade], color=cor, width=0.4)
        ax1.axhline(0, color='black', linewidth=1)
        ax1.set_ylim(-1.1, 1.1)
        ax1.set_ylabel("Score (-1 a +1)")
        ax1.set_title(f"Sentimento Geral: {resultado} {emoji}")
        ax1.grid(axis='y', linestyle='--', alpha=0.5)
        va = 'bottom' if polaridade >= 0 else 'top'
        offset = 0.05 if polaridade >= 0 else -0.05
        ax1.text(0, polaridade + offset, f'{polaridade:.2f}', ha='center', va=va, fontweight='bold')

        categorias = ['Negativo', 'Neutro', 'Positivo']
        valores = [scores['neg'], scores['neu'], scores['pos']]
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
st.caption("Desenvolvido por Andressa Gabriel · [GitHub](https://github.com/Andressagabriell)")
