# Implémentez un modèle de scoring
## Présentation:
L'entreprise Prêt a dépenser, société financière, propose des crédits à la consommation pour des personnes ayant peu ou
pas du tout d'historique de prêt. 
Cette entreprise souhaite mettre en oeuvre un outil de "scoring crédit" pour calculer la probabilité qu'un client 
rembourse son crédit 
puis classifie la demande en crédit accordé ou refusé.
Elle souhaite développer un algorithme de classification en s'appuyantsur des sources de données variées (donées 
comportementales, données provenant d'autres institutions financières, etc)

De plus, les chargés de relation clients ont fait remonter le fait que les clients sont de plus en plus demandeurs de
transparence vis-à-vis des décisions d'octroi de crédit.
Cette demande de transparence des clients va tout à fait dans le sens des valeurs que l'entreprise veut incarner.

Prêt à dépenser décide donc de développer un dashboard interactif pour que les chargés de relation client puissent à 
la fois expliquer de façon la plus transparente possible les décisions d'octroi de crédit, mais également permettre à
leurs clients de disposer de leurs informations personnelles et de les explorer facilement.

## Construction:
Dans ce dépôt, vous trouverez plusieurs fichiers:

- "script.py" : préparation des données, entraînement et configuration du modèle de classification
(KNeighbors Classifier).
- "app.py" : permet de générer le dashboard interactif.

## Données: 
- Les données ont été téléchargés à cette adresse : https://www.kaggle.com/c/home-credit-default-risk/data

- Kernel Kaggle utilisé pour facilité l'analyse exploratoire, la préparation des données et le feature engineering 
nécessaire à l'élaboration du modèle de scoring : https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features/script

## Packages: 
Différents packages Python ont été utilisés: 

- scikit-learn 1.0.2
- smote 0.1
- numpy 1.21.5
- mlflow 2.1.1
- xgboost 1.7.4
- pandas 1.4.4
- numpy 1.21.5
- lightgbm 3.3.5
- scipy 1.9.1
- joblib 1.2.0
- shap 0.41.0
- streamlit 1.20.0
- requests 2.28.1
- plotly 5.9.0
- matplotlib 3.5.2
- json 2.0.9
- evidently 0.2.8

