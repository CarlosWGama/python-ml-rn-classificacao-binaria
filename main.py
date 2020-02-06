import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

#Separa os dados
csv = pd.read_csv('resfriado.csv', sep=',')
dados = csv.values
atributos = dados[:,1:]
classificadores = dados[:,0]

#Separando
aTre, aTes, cTre, cTes = train_test_split(atributos, classificadores, test_size=0.3)

#Carrega um modelo existente
# from keras.models import load_model
# modelo = load_model('modelo.h5')

#Cria o modelo
modelo = Sequential()
modelo.add(Dense(units=5, activation='sigmoid', input_dim=8))
modelo.add(Dense(units=1, activation='sigmoid'))
modelo.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
#Treinando
modelo.fit(aTre, cTre, batch_size=10, epochs=500)

#Salva o modelo (Opcional)
modelo.save('modelo.h5') 

#Avalia
resultados = modelo.predict(aTes)
resultados = list(map(lambda r: 1 if r > 0.5 else 0, resultados))
#Cria um vetor True ou False quando acerta um valor
comparacao = resultados == cTes #Retorna True quando valores são iguais

print('Total:', len(comparacao), '| Acertos: ', np.sum(comparacao))