"""
sentiment_core.py
------------------
Lógica central de tradução + análise de sentimento (VADER).

Este módulo NÃO depende de terminal (input/print) nem de gráfico (matplotlib).
Isso permite reutilizá-lo em:
  - CLI interativo (main.py)
  - Processamento em lote via Excel (excel_batch.py)
  - Uma futura API (app.py / FastAPI)
  - Integração com CRM

Mantendo uma única fonte de verdade para a lógica de análise.
"""

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator

# ---------------------------------------------------------------------------
# Setup (executado uma vez, na importação do módulo)
# ---------------------------------------------------------------------------

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Reutiliza a mesma instância do analisador em todas as chamadas
# (evita recriar o objeto a cada frase — importante em lote/API).
_analyzer = SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------------
# Exceções específicas do domínio
# ---------------------------------------------------------------------------

class TraducaoIndisponivelError(Exception):
    """Levantada quando a tradução falha e não há fallback seguro."""
    pass


# ---------------------------------------------------------------------------
# Funções principais
# ---------------------------------------------------------------------------

def traduzir_para_ingles(texto: str, lancar_erro: bool = False) -> str:
    """
    Traduz um texto em português para inglês (o VADER só entende inglês).

    Args:
        texto: texto original em português.
        lancar_erro: se True, propaga TraducaoIndisponivelError em caso de
                     falha. Se False (padrão), retorna o texto original
                     sem tradução (comportamento de fallback, útil no CLI).

    Returns:
        Texto traduzido para inglês (ou o texto original, se lancar_erro=False
        e a tradução falhar).
    """
    texto = (texto or "").strip()
    if not texto:
        return ""

    try:
        return GoogleTranslator(source='pt', target='en').translate(texto)
    except Exception as e:
        if lancar_erro:
            raise TraducaoIndisponivelError(f"Falha ao traduzir: {e}") from e
        return texto


def classificar_score(scores: dict, limite: float = 0.35, limite_neutro: float = 0.65) -> str:
    """
    Classifica o resultado do VADER em Positivo/Negativo/Neutro.

    Combina duas regras para reduzir falsos positivos/negativos em frases
    fracamente polarizadas (ex: "caixa original" empurrando uma frase
    puramente descritiva para "Positivo" por causa de 1 palavra):

    1) Threshold mais alto que o padrão do VADER (0.05): por padrão, só
       classifica como Positivo/Negativo se compound >= 0.35 (ou <= -0.35).
       O padrão do VADER é sensível demais a ruído de 1 palavra isolada.

    2) Predomínio de neutralidade: se a maior parte da frase (>= 65%) foi
       classificada como neutra pelo próprio VADER, e o compound não é
       fortemente polarizado (< 0.35 em módulo), força "Neutro".

    Valores calibrados testando frases reais de e-commerce/atendimento
    (ex: "chegou na caixa original" não deve virar "Positivo" só por
    causa da palavra "original").

    Args:
        scores: dict com as chaves 'neg', 'neu', 'pos', 'compound'
                (mesmo formato retornado por analyzer.polarity_scores()).
        limite: threshold mínimo (em módulo) do compound para considerar
                Positivo/Negativo. Ajustável conforme o domínio.
        limite_neutro: proporção mínima de 'neu' para acionar a regra 2.

    Returns:
        "Positivo" | "Negativo" | "Neutro"
    """
    compound = scores["compound"]

    # Regra 2: frase majoritariamente neutra e sem polarização forte
    if scores["neu"] >= limite_neutro and abs(compound) < 0.3:
        return "Neutro"

    # Regra 1: threshold padrão (mais conservador que o 0.05 "cru" do VADER)
    if compound >= limite:
        return "Positivo"
    elif compound <= -limite:
        return "Negativo"
    return "Neutro"


def analisar_texto(texto_pt: str, lancar_erro_traducao: bool = False) -> dict:
    """
    Pipeline completo: traduz um texto em português e retorna a análise
    de sentimento do VADER.

    Args:
        texto_pt: frase original em português.
        lancar_erro_traducao: repassado para traduzir_para_ingles().

    Returns:
        dict com as chaves:
            texto_original   -> str
            texto_traduzido  -> str
            neg, neu, pos    -> float (0 a 1, somam 1.0)
            compound         -> float (-1 a 1)
            classificacao    -> "Positivo" | "Negativo" | "Neutro"
    """
    texto_traduzido = traduzir_para_ingles(texto_pt, lancar_erro=lancar_erro_traducao)
    scores = _analyzer.polarity_scores(texto_traduzido)

    return {
        "texto_original": texto_pt,
        "texto_traduzido": texto_traduzido,
        "neg": scores["neg"],
        "neu": scores["neu"],
        "pos": scores["pos"],
        "compound": scores["compound"],
        "classificacao": classificar_score(scores),
    }


def analisar_lote(textos: list[str]) -> list[dict]:
    """
    Analisa uma lista de textos em português, um por um.

    Não interrompe o lote inteiro se uma tradução falhar: registra o erro
    naquela linha e continua com as demais (importante para processar
    planilhas grandes sem perder todo o trabalho por causa de 1 linha ruim).
    """
    resultados = []
    for texto in textos:
        try:
            resultados.append(analisar_texto(texto))
        except Exception as e:
            resultados.append({
                "texto_original": texto,
                "texto_traduzido": "",
                "neg": None, "neu": None, "pos": None, "compound": None,
                "classificacao": f"ERRO: {e}",
            })
    return resultados