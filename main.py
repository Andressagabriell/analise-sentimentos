"""
main.py
-------
Modo interativo (CLI): digita uma frase, vê o resultado no terminal
e um gráfico de sentimento (geral + composição).
"""

import matplotlib.pyplot as plt
from sentiment_core import analisar_texto


EMOJIS = {"Positivo": "😊", "Negativo": "😠", "Neutro": "😐"}
CORES = {"Positivo": "green", "Negativo": "red", "Neutro": "gray"}


def plotar_resultado(resultado: dict):
    """Gera o gráfico de 2 painéis: sentimento geral + composição."""
    compound = resultado["compound"]
    classificacao = resultado["classificacao"]
    cor = CORES[classificacao]
    emoji = EMOJIS[classificacao]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Painel 1: escala -1 a 1
    ax1.bar(['Sentimento'], [compound], color=cor, width=0.4)
    ax1.axhline(0, color='black', linewidth=1)
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_ylabel("Score (-1 = muito negativo, +1 = muito positivo)")
    ax1.set_title(f"Sentimento Geral: {classificacao} {emoji}")
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    va = 'bottom' if compound >= 0 else 'top'
    offset = 0.05 if compound >= 0 else -0.05
    ax1.text(0, compound + offset, f'{compound:.2f}', ha='center', va=va, fontweight='bold')

    # Painel 2: composição neg/neu/pos
    categorias = ['Negativo', 'Neutro', 'Positivo']
    valores = [resultado["neg"], resultado["neu"], resultado["pos"]]
    ax2.bar(categorias, valores, color=['red', 'gray', 'green'], width=0.5)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Proporção (soma = 1.0)")
    ax2.set_title("Composição do Sentimento")
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    for i, v in enumerate(valores):
        ax2.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')

    fig.suptitle(f'Frase analisada: "{resultado["texto_original"]}"', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


def main():
    print("--- Termômetro de Sentimentos VADER ---")
    texto = input("Digite uma frase para análise: ")

    if not texto.strip():
        print("Erro: Digite algum texto.")
        return

    resultado = analisar_texto(texto)

    print(f"\nTexto traduzido (uso interno): {resultado['texto_traduzido']}")
    print(f"Resultado: {resultado['classificacao']}")
    print(f"Score (compound): {resultado['compound']:.4f} (Varia de -1 a 1)")
    print(f"Negativo: {resultado['neg']:.2f} | Neutro: {resultado['neu']:.2f} | Positivo: {resultado['pos']:.2f}")

    plotar_resultado(resultado)


if __name__ == "__main__":
    main()