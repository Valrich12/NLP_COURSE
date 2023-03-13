#Importamos la biblioteca spacy
import pandas as pd
import spacy
from spacy import displacy

def separate_news(file_name):

    # Lee el corpus original del archivo de entrada y lo pasa a una DataFrame
    df = pd.read_csv(file_name, sep='&&&&&&&&', dtype=str,engine='python')
    news = df.drop(df.columns[[0,1,3]],axis=1)
    print(news)
    news = news.to_string(index=False)
    
    f = open('test.txt','w',encoding='utf-8')
    f.write(news)
    f.close
    
    return news



#conda install -c conda-forge spacy-model-es_core_news_sm
#python -m spacy download es_core_news_sm

if __name__ == "__main__":
    news = separate_news('corpus_noticias.txt')
    print("ESTO ES UNA PRUEBA")

# cadena = "Juan estaba corriendo por el pasillo de la escuela superior de cómputo. "
# # ~ cadena = "Los perros ladraron la otra noche a unos coches rojos que pasaron por la calle."

# #Se carga el corpus para el tagger en español
    # nlp = spacy.load("es_core_news_sm")
# #Se realiza el análisis de la cadena
    # doc = nlp(news)

    # for token in doc:
    #     print(token.text, token.pos_, token.dep_, token.lemma_)
    #     print(token.text, token.pos_, token.dep_, token.lemma_, spacy.explain(token.tag_), spacy.explain(token.dep_))
    # displacy.serve(doc, style="dep")    

# for entity in doc.ents:
#     print(entity.text, entity.label_)
