import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import json
import random

#################### Charger le model #############################
from keras.models import load_model
model = load_model('model_Sequential.h5')

###################################################################
with open('data.json', encoding='utf-8') as data:
    intents = json.load(data)
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

###################################################################
def clean_up_sentence(sentence):
    # tokeniser le pattern - diviser les mots en tableau
    sentence_words = nltk.word_tokenize(sentence)
    # radical chaque mot - créer une forme courte pour le mot
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# retourner un tableau : 0 ou 1 pour chaque mot qui existe dans la phrase

def bow(sentence, words, show_details=True):
    # tokeniser  le  pattern
    sentence_words = clean_up_sentence(sentence)
    # tableau de mots - matrice de N mots, matrice de vocabulaire
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # attribuer 1 si le mot actuel est dans la position du vocabulaire
                bag[i] = 1
                if show_details:
                    print ("bag trouvé : %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filtrer les prédictions en dessous d'un seuil
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # trier par force de probabilité
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list



from urllib.parse import *

parse_url = urlparse('http://127.0.0.1:8000/form/')

def getResponse(ints, intents_json):
    tag = ints[0]['intent']  
    list_of_intents = intents_json['intents']
    x = urlunparse(parse_url)
    for i in list_of_intents:
        if(i['tag']== "covid"):
            liste = [x]
            i = 0
            for i in range(len(liste)):
                result = liste[i]       
        elif(i['tag']== tag):
            print(10*"@",tag)
            result = random.choice(i['responses'])
            break
    return result

def reponse_chatbot(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


def getSymptomes():
    liste = ["avec vous de la fievre ?","vous toussez ?","Avez vous plus de 60 ans"]
    for i in range(len(liste)):
        msg  = liste[i]



############ FLASK APP ############
from flask import Flask, render_template, request
from joblib import dump, load

model_dtree = load('DecisionTree.joblib')

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")
    
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return reponse_chatbot(userText)

@app.route('/form/')
def form():
    return render_template('form.html')

@app.route('/data/', methods = ['POST', 'GET'])
def data():
    toux = request.form['toux']
    fievre = request.form['fievre']
    gorge = request.form['gorge']
    tete = request.form['tete']
    essoufle = request.form['essoufle']
    age = request.form['age']
    liste_symptomes = []

    liste_symptomes.append(toux)
    liste_symptomes.append(fievre)
    liste_symptomes.append(gorge)
    liste_symptomes.append(tete)
    liste_symptomes.append(essoufle)
    liste_symptomes.append(age)

    symptomes = [
        "Bonjour,  voici le résultat de votre test:     ", 
        "Pour 'la toux', vous avez répondu : ", liste_symptomes[0],
        "Pour 'la fièvre', vous avez répondu : ", liste_symptomes[1], 
        "Pour 'le mal à la gorge' vous avez répondu : ", liste_symptomes[2],
        "Pour 'le mal à la tête', vous avez répondu : ", liste_symptomes[3],
        "Pour 'essouflé', vous avez répondu : " , liste_symptomes[4],
        "Pour 'age de plus de 60 ans', vous avez répondu : " , liste_symptomes[5]
        
        ]

    liste_sym = liste_symptomes
    for i in range(len(liste_sym)):
        if (liste_sym[i] == "oui"):
            liste_sym[i] = 1
        else:
            liste_sym[i] = 0

    print(liste_sym)

    prediction = model_dtree.predict([liste_sym])

   
    test =''
    if prediction[0] == 1:   
        test = "Vous êtes positif au covid 19 "
    else:
        test = "Vous êtes négatif au covid 19 "
    return render_template('data.html',symptomes = symptomes,  message =  test  )   


if __name__ == "__main__":
    app.run(port=8000)