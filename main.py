import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
import matplotlib.pyplot as plt

# Garante que o léxico do VADER esteja baixado
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')


def traduzir_para_ingles(texto):
    """
    O VADER só entende inglês. Como o usuário digita em português,
    traduzimos o texto antes de analisar.
    """
    try:
        traduzido = GoogleTranslator(source='pt', target='en').translate(texto)
        return traduzido
    except Exception as e:
        print(f"Aviso: falha na tradução ({e}). Analisando o texto original.")
        return texto


def analisar_sentimento_vader(texto_em_ingles):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(texto_em_ingles)
    return scores  # retorna o dicionário completo: neg, neu, pos, compound


def main():
    print("--- Termômetro de Sentimentos VADER ---")
    texto = input("Digite uma frase para análise: ")

    if not texto.strip():
        print("Erro: Digite algum texto.")
        return

    # 1) Traduz a frase em português para inglês
    texto_traduzido = traduzir_para_ingles(texto)
    print(f"Texto traduzido (uso interno): {texto_traduzido}")

    # 2) Analisa o sentimento sobre o texto já em inglês
    scores = analisar_sentimento_vader(texto_traduzido)
    polaridade = scores['compound']

    # Define cor e rótulo baseado no score
    if polaridade >= 0.05:
        resultado = "Positivo"
        cor = 'green'
        emoji = "😊"
    elif polaridade <= -0.05:
        resultado = "Negativo"
        cor = 'red'
        emoji = "😠"
    else:
        resultado = "Neutro"
        cor = 'gray'
        emoji = "😐"

    print(f"\nResultado: {resultado}")
    print(f"Score (compound): {polaridade:.4f} (Varia de -1 a 1)")
    print(f"Negativo: {scores['neg']:.2f} | Neutro: {scores['neu']:.2f} | Positivo: {scores['pos']:.2f}")

    # ---------- Geração do Gráfico (2 painéis lado a lado) ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Painel 1: Escala geral de -1 a 1 (direção + intensidade) ---
    ax1.bar(['Sentimento'], [polaridade], color=cor, width=0.4)
    ax1.axhline(0, color='black', linewidth=1)  # linha do zero
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_ylabel("Score (-1 = muito negativo, +1 = muito positivo)")
    ax1.set_title(f"Sentimento Geral: {resultado} {emoji}")
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    # Valor em cima ou embaixo da barra, dependendo do sinal
    va = 'bottom' if polaridade >= 0 else 'top'
    offset = 0.05 if polaridade >= 0 else -0.05
    ax1.text(0, polaridade + offset, f'{polaridade:.2f}',
              ha='center', va=va, fontweight='bold')

    # --- Painel 2: Composição (negativo / neutro / positivo) ---
    categorias = ['Negativo', 'Neutro', 'Positivo']
    valores = [scores['neg'], scores['neu'], scores['pos']]
    cores_barras = ['red', 'gray', 'green']

    ax2.bar(categorias, valores, color=cores_barras, width=0.5)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Proporção (soma = 1.0)")
    ax2.set_title("Composição do Sentimento")
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    for i, v in enumerate(valores):
        ax2.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')

    fig.suptitle(f'Frase analisada: "{texto}"', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # reserva espaço no topo para o suptitle
    plt.show()


if __name__ == "__main__":
    main()