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
    return scores['compound']


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
    polaridade = analisar_sentimento_vader(texto_traduzido)

    # Define cor e rótulo baseado no score
    if polaridade >= 0.05:
        resultado = "Positivo"
        cor = 'green'
    elif polaridade <= -0.05:
        resultado = "Negativo"
        cor = 'red'
    else:
        resultado = "Neutro"
        cor = 'gray'

    print(f"\nResultado: {resultado}")
    print(f"Score: {polaridade:.4f} (Varia de -1 a 1)")

    # Geração do Gráfico
    plt.figure(figsize=(8, 5))
    bars = plt.bar(['Intensidade'], [abs(polaridade)], color=cor, width=0.4)

    # Adiciona o valor exato em cima da barra
    plt.text(0, abs(polaridade) + 0.02, f'{abs(polaridade):.2f}',
              ha='center', va='bottom', fontweight='bold')

    plt.title(f"Análise de Sentimento: {resultado}", fontsize=14)
    plt.ylabel("Intensidade (0 a 1)", fontsize=12)
    plt.ylim(0, 1.1)  # Espaço extra para o texto não cortar
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()