from os import PathLike
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump
import pandas as pd
import pathlib

#ruta
df = pd.read_csv(pathlib.Path('data/pishing-dataset.csv'))

#quitar espacios
df.columns = df.columns.str.strip()

#separar feautures y targets
y = df.pop('Result')
X = df

#dividir train/test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

#entrenando modelo randomforest
print ('Training model.. ')
clf = RandomForestClassifier(n_estimators = 10,
                            max_depth=2,
                            random_state=0)
clf.fit(X_train, y_train)

#evaluacion rapida
y_pred = clf.predict(X_test)
print(f" Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))


#guardar el modelo
print ('Saving model..')

dump(clf, pathlib.Path('model/pishing-dataset.joblib'))


