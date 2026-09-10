# QwAn - Parlement NLP

Dépôt du projet QwAN visant à entraîner un modèle d'embedding sensible à l'opinion (_stance aware_) en fine-tunant un Qwen3-Embedding-0.6B sur un corpus de débats parlementaires.

Le modèle est disponible [sur Hugging Face](https://huggingface.co/medialab-sciencespo/QwAn), ainsi que les [jeux de données utilisées](https://huggingface.co/datasets/medialab-sciencespo/QwAn-dataset).

## Structure

Le projet se divise en plusieurs parties :

* la collecte et traitement des débats de l'Assemblée nationale puis la constitution du corpus d'entraînement (`/data`)
* l'entraînement de notre modèle (`/training`)
* son évaluation (`/evaluation`)
* le cas d'application (`/application`)
