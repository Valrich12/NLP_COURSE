from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import spacy

#Se define una lista de stop words, de acuerdo a lo que se pide en la práctica
stop_words = { ##Articulos
                        'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
                        'lo', 'al', 'del', 'este', 'ese', 'aquel', 'un poco de', 'mucho', 'poco',
                        'otro', 'cierto', 'algún', 'alguna', 'algunos', 'algunas', 'varios', 'varias',
                        'ambos', 'ambas', 'cada', 'cualquier', 'cualquieras', 'demasiado', 'demasiada', 
                        'demasiados', 'demasiadas', 'menos', 'más', 'medio', 'media', 'medios', 'medias',
                        'ningún', 'ninguna', 'ningunos', 'ningunas', 'varios', 'varias', 'poco', 'poca', 'a la',
                    ##Preposiciones
                        'a', 'ante', 'bajo', 'cabe', 'con', 'contra', 'de', 'desde', 'durante', 'en', 'entre',
                        'hacia', 'hasta', 'mediante', 'para', 'por', 'segun', 'sin', 'so', 'sobre', 'tras',
                        'versus', 'vía', 'a través de', 'a causa de', 'a pesar de', 'a propósito de', 'a raíz de',
                        'durante', 'excepto', 'frente a', 'junto a', 'menos', 'salvo', 'según', 'según con',
                        'sobre todo', 'dentro de', 'encima de', 'detrás de', 'fuera de', 'más allá de', 'debajo de',
                        'dentro de', 'de', 'DE',
                    ##Conjunciones
                        'y', 'e', 'ni', 'o', 'u', 'que', 'si', 'mas', 'pero', 'aunque', 'sino', 'para que', 'porque',
                        'ya que', 'pues', 'como', 'así que', 'mientras', 'cuando', 'después', 'antes', 'hasta que',
                        'siempre que', 'a menos que', 'en caso de que', 'con tal de que', 'sin que', 'por más que',
                        'a fin de que', 'a pesar de que', 'en tanto que', 'aunque no', 'por cuanto', 'sea que',
                        'de manera que', 'por lo tanto', 
                    ##Pronombres
                        'yo', 'tu', 'el', 'ella', 'usted', 'nosotros', 'nosotras', 'vosotros', 'vosotras', 'ellos',
                        'ellas', 'ustedes', 'quien', 'quienes', 'cual', 'cuales', 'cuanto', 'cuanta', 'cuantos',
                        'cuantas', 'que', 'esto', 'eso', 'aquello', 'nada', 'algo', 'alguien', 'nadie', 'quienquiera',
                        'quienesquiera', 'cualquiera', 'cualesquiera', 'cuantoquiera', 'cuantaquiera', 'cuantosquiera',
                        'cuantasquiera', 'quequiera', 'dondequiera', 'comoquiera', 'cuandoquiera', 'estos', 'esos',
                    ##Otros
                        'edd'
                    }

def cosine_similarity(x, y):
    x = np.ravel(x)
    y = np.ravel(y)
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    similarity = dot_product / (norm_x * norm_y)
    return similarity

# Leemos el archivo que contiene el corpus de noticias 
with open("corpus_noticias_procesado.txt", "r") as f:
    noticias = f.read()
df = pd.DataFrame(noticias.splitlines())

# Representación vectorial binarizada
vectorizador_binario = CountVectorizer(binary=True, token_pattern= r'\b\w+\b')
X_binario = vectorizador_binario.fit_transform(df[0])
print (vectorizador_binario.get_feature_names_out())
print ('Representación vectorial binarizada')
print (X_binario.toarray())#dense ndarray

# Representación vectorial por frecuencia
vectorizador_frecuencia = CountVectorizer(token_pattern= r'\b\w+\b')
X_frecuencia = vectorizador_frecuencia.fit_transform(df[0])
print('Representación vectorial por frecuencia')
print (X_frecuencia.toarray())

# Representación vectorial tf-idf
vectorizador_tfidf = TfidfVectorizer(token_pattern= r'\b\w+\b')
X_tfidf = vectorizador_tfidf.fit_transform(df[0])
print ('Representación vectorial tf-idf')
print (X_tfidf.toarray())

#Se pide el nombre del archivo a procesar
archivoAProcesar = input("Ingrese el nombre del archivo a procesar, incluyendo la extensión .txt=")

# Leemos el archivo que contiene la noticia a ser comparada
with open(archivoAProcesar, "r") as f:
    noticia = f.read().splitlines()
df_noticia = pd.DataFrame(noticia)

#Se carga el corpus para el tagger en español
nlp = spacy.load('es_core_news_sm')

# Tokenizar, lematizar y remover stop words en la noticia
noticias_procesadas = []
for noticia in df_noticia[0]:
    doc = nlp(noticia)
                                                                 #and not token.is_punct: signos de puntación
    tokens = [token.lemma_ for token in doc if token.lemma_.lower() not in stop_words]
    noticias_procesadas.append(" ".join(tokens))

# Crear un nuevo archivo con el corpus procesado
with open('noticia_procesada.txt', 'w', encoding='utf-8') as file:
    for noticia in noticias_procesadas:
        file.write(noticia + '\n')

# Leemos el archivo que contiene la noticia a ser comparada
with open("noticia_procesada.txt", "r") as f:
    noticia = f.read().splitlines()
df_noticia = pd.DataFrame(noticia)

# Se pide el tipo de vectorización que desea aplicar a la nueva noticia.
tipo_vectorizacion = input("Elija el tipo de vectorización que desea aplicar a la nueva noticia (frecuencia, binaria o tf-idf): ")

# Crear un vectorizador con el tipo de vectorización elegido por el usuario.
if tipo_vectorizacion == "frecuencia":
    vectorizador = vectorizador_frecuencia.transform([df_noticia.iloc[0][0]])
elif tipo_vectorizacion == "binaria":
    vectorizador = vectorizador_binario.transform([df_noticia.iloc[0][0]])
elif tipo_vectorizacion == "tf-idf":
    vectorizador = vectorizador_tfidf.transform([df_noticia.iloc[0][0]])
else:
    print("Tipo de vectorización no válido. Por favor, seleccione 'frecuencia', 'binaria' o 'tfidf'.")
    exit()

# Vectorizar la nueva noticia
nueva_noticia_vectorizada = vectorizador

# Calcular la similitud coseno entre la nueva noticia y todas las noticias en el dataframe
similitudes = []
for i in range(df.shape[0]):
    if tipo_vectorizacion == "frecuencia":
        noticia_vectorizada = vectorizador_frecuencia.transform([df.iloc[i][0]])
    elif tipo_vectorizacion == "binaria":
        noticia_vectorizada = vectorizador_binario.transform([df.iloc[i][0]])
    elif tipo_vectorizacion == "tf-idf":
        noticia_vectorizada = vectorizador_tfidf.transform([df.iloc[i][0]])
    similitud = cosine_similarity(nueva_noticia_vectorizada.toarray(), noticia_vectorizada.toarray())
    noticia_num = df.index[i]
    similitudes.append((similitud, noticia_num))

# Agregar la columna de similitudes al dataframe
df["similitud"] = [sim[0] for sim in similitudes]
df["noticia_num"] = [sim[1] for sim in similitudes]

# Ordenar el dataframe por similitud descendente y mostrar las 10 noticias más similares
df = df.sort_values(by=["similitud"], ascending=False).reset_index(drop=True)
print(df.head(10))
