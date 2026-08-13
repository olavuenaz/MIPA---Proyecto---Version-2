from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json, numpy as np

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
anchors = json.load(open("intent_anchors.json"))
crisis_anchor = model.encode(anchors["CRISIS_ALERTA"]).mean(axis=0).reshape(1,-1)

no_relacionadas = ["qué frío hace hoy", "quiero una galleta", "hola cómo estás",
                    "explícame cómo funciona el motor", "me gusta el fútbol",
                    "vamos al cine este fin de semana", "qué hora es"]
relacionadas = ["quiero morir", "no puedo más con esto", "pensé en quitarme la vida",
                 "ya no le veo sentido a seguir", "he pensado en hacerme daño"]

for grupo, frases in [("NO relacionadas", no_relacionadas), ("SÍ relacionadas", relacionadas)]:
    scores = [cosine_similarity(model.encode([f]), crisis_anchor)[0][0] for f in frases]
    print(f"{grupo}: media={np.mean(scores):.3f}  min={np.min(scores):.3f}  max={np.max(scores):.3f}")