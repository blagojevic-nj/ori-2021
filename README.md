# Klasifikacija rasa pasa na osnovu slike
Projekat iz predmeta Osnove rečunarske inteligencije za školsku 2021 godinu. 

## Opis problema
Projekat se bavi rešavanjem problema identifikacije rase psa na osnovu slike psa. Rešenje je implementirano uz pomoć konvolucijskih neuronskih mreža iz keras python biblioteke.

## Dataset
Dataset koji je korišćen za treniranje je preuzet sa sledećeg linka: [dataset](https://www.kaggle.com/c/dog-breed-identification/data) koji se sastoji od 10000 slika i 120 različitih rasa. Da bismo pojednostavili projekat, uzete su slike 10 rasa pasa koje imaju najviše slika i nad njima je vršeno treniranje.

## Potrebne biblioteke
- Tensorflow
- Keras
- sklearn
- pandas
- numpy
- matplotlib
- Pillow
- PyQt5

## Pokretanje
Potrebno je klonirati repozitorijum i skinuti dataset. Zatim je potrebno raspakovati dataset i staviti u root folder projekta.
Nakon toga potrebno je otvoriti komandnu liniju u root folderu projekta i uneti sledece.
```sh 
python model.py
```
čime se pokreće kreiranje baze podataka i treniranje modela.

Za kraj, nakon što se istrenira model, praktično možete testirati rešenje pokretanjem
```sh 
python viewer.py
```
