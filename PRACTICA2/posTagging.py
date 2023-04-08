# Importamos la biblioteca spacy
import re
import pandas as pd
import spacy
from spacy import displacy


def separate_news(file_name):

    # Lee el corpus original del archivo de entrada y lo pasa a una DataFrame
    df = pd.read_csv(file_name, sep='&&&&&&&&', dtype=str, engine='python')
    news = df.drop(df.columns[[0, 1, 3]], axis=1)
    news = news.to_string(index=False)
    news = news.replace("  ", "")
    news = re.sub(r"^\s+|\s+$", r"", news, flags=re.MULTILINE)
    return news


# conda install -c conda-forge spacy-model-es_core_news_sm
# python -m spacy download es_core_news_sm
if __name__ == "__main__":
    news = separate_news('corpus_noticias.txt')

# cadena = "Juan estaba corriendo por el pasillo de la escuela superior de cómputo. "
# # ~ cadena = "Los perros ladraron la otra noche a unos coches rojos que pasaron por la calle."

# #Se carga el corpus para el tagger en español
    modnlp = spacy.util.get_lang_class('es')
    modnlp.Defaults.stop_words = {"un", "una", "unos", "unas", "el", "los", "la", "las", "lo", "yo", "mi", "conmigo", "me", "nosotros", "nosotras", "nos", "tú", "usted", "ti", "contigo", "ustedes", "él", "ella", "ello", "si", "consigo", "lo", "a", "ante",
                                  "bajo", "con", "contra", "de", "desde", "durante", "en", "entre", "hacia", "hasta", "mediante", "para", "por", "según", "sin", "sobres", "tras", "ni", "tanto", "como", "o", "bien", "y", "sino", "también", "pero", "ya", "para", "mientras", "luego"}
    stopWords = modnlp.Defaults.stop_words
    nlp = spacy.load("es_core_news_sm")

# # #Se realiza el análisis de la cadena
    nlp.max_length = 1475807
    doc = nlp(news)
    t = open('tokens.txt', 'a', encoding='utf-8')
    l = open('lema.txt', 'a', encoding='utf-8')
    for token in doc:
        if not token.is_stop:
            if token.text == '\n' :
                t.write(token.text)
                l.write(token.lemma_)
            else:
                t.write(token.text+" ")
                l.write(token.lemma_+" ")
        # print(token.text, token.pos_, token.dep_, token.lemma_, spacy.explain(token.tag_), spacy.explain(token.dep_))
    l.close
    t.close

# for entity in doc.ents:
#     print(entity.text, entity.label_)
