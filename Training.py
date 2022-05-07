import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
#from keras.optimizers import SGD
from tensorflow.keras.optimizers import SGD
import random

words=[]
classes = []
documents = []
ignore_words = ['?', '!']

with open('data.json', encoding='utf-8') as data:
    intents = json.load(data)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize chaque mot
        w = nltk.word_tokenize(pattern)
        words.extend(w)    
        documents.append((w, intent['tag']))
        # Ajouter à notre liste de classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmaztize et mettre en minscule chaque mot + retirer les doublons
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))
# documents = conbinaison de  patterns +  intents
print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = tous les mots,  vocabulaire
print (len(words), "racine mots uniquess", words)

pickle.dump(words,open('texts.pkl','wb'))
pickle.dump(classes,open('labels.pkl','wb'))

#####################""
training = []
# array pour les output
output_empty = [0] * len(classes)
# training set, bag of words de chaque phrase
for doc in documents:
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize chaque mot - créer un mot de base, dans le but de représenter des mots liés
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # créer notre tableau de mots avec 1, 
    # si la correspondance de mots est trouvée dans le modèle actuel
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # output est un '0' pour chaque tag 
    # et un '1' pour la tag actuelle (pour chaque pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1    
    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")


# 3 layers. 1er layer 128 neurons, 2eme  64 neurons et  3eme  output layer contient le nombre de neurons
# égale au nombre de "intents"  à  prédire  dans output intent avec  softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compilation du modele. 
# Stochastic gradient descent avec  Nesterov accelerated gradient donnes des bons  résultats pour ce modèle
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# fitting et sauvegarde du modèle
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('model_Sequential.h5', hist)
print("model created")