# Analisador de Sentimentos VADER (PT-BR)

Ferramenta de análise de sentimentos em português que combina a precisão do léxico **VADER** (NLTK) com tradução neural via **Google Translator**.

## 🚀 Por que esta abordagem?

O VADER é extremamente eficiente para redes sociais e textos curtos, mas foi treinado exclusivamente em inglês. Em vez de usar modelos pesados ou treinar do zero, este projeto utiliza uma camada de **tradução automática** para converter o input em português para inglês antes da análise.

Isso garante:
- **Alta precisão** sem necessidade de grandes datasets de treinamento.
- **Leveza:** Roda localmente com poucas dependências.
- **Visualização rica:** Gráficos de intensidade e composição emocional.

## 📦 Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/Andressagabriell/analise-sentimentos.git
   cd analise-sentimentos
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Execute o programa:
   ```bash
   python main.py
   ```

## 👩‍💻 Autora

**Andressa Gabriel**
📎 GitHub: [github.com/Andressagabriell](https://github.com/Andressagabriell)
🔗 LinkedIn: _em breve_