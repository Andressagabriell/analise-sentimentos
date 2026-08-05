import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Garante que o léxico do VADER esteja baixado
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

def analisar_sentimento_vader(texto):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(texto)
    return scores['compound']

def main():
    print("--- Termômetro de Sentimentos VADER ---")
    texto = input("Digite uma frase para análise: ")
    
    if not texto.strip():
        print("Erro: Digite algum texto.")
        return

    polaridade = analisar_sentimento_vader(texto)
    
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

    # Geração do Gráfico Corrigida
    plt.figure(figsize=(8, 5))
    bars = plt.bar(['Intensidade'], [abs(polaridade)], color=cor, width=0.4)
    
    # Adiciona o valor exato em cima da barra
    plt.text(0, abs(polaridade) + 0.02, f'{abs(polaridade):.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title(f"Análise de Sentimento: {resultado}", fontsize=14)
    plt.ylabel("Intensidade (0 a 1)", fontsize=12)
    plt.ylim(0, 1.1) # Espaço extra para o texto não cortar
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.show()

if __name__ == "__main__":
    main()
