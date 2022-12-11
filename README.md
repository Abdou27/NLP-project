# Projet NLP
* **Étudiants :** Abderrahim BENMELOUKA, Maxime VINCENT
* **Formation :** Master 2 ISD Université Paris-Saclay

Le but de ce projet est d'appliquer différentes méthodes de NLP
à un dataset d'actualités et de les comparer entre elles.

## Tâches à réaliser
* Analyser et visualiser les données
* Utiliser différentes méthodes de pré-processing notamment de vectorisation
* Appliquer des modèles de classification mutli-classe supervisés
* Appliquer des modèles de clustering non-supervisés en fixant 
le nombre de clusters au nombre de classes
* Utiliser différentes méthodes d'évaluation des modèles
* Utiliser des graphiques pour bien illustrer les résultats
* Commenter la méthodologie utilisée
* Conclure sur les résultats obtenus

## Environnement d'exécution
En principe, il n'est pas nécessaire de relancer notre code car les notebook
sont disponibles au format HTML.

Mais si vous voulez tout de même les exécuter alors il peut être utile
de construire un environnement virtuel pour éviter de polluer l'environnement Python global.
Notez que le nombre de librairies utilisées étant relativement important,
le temps de téléchargement peut être conséquent.
L'environnement virtuel final pèse environ 4Go.
```shell
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
jupyter-notebook
```

Vous pouvez ensuite sortir de l'environnement virtuel avec `deactivate`.
