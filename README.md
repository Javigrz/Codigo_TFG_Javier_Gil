# Código TFG Javier Gil
Este repositorio contiene el código fuente del Trabajo de Fin de Grado de Javier Gil.

## Descripción
El proyecto desarrolla un sistema integral de análisis de sentimientos aplicable en tiempo real a datos de redes sociales. Abarca desde la recopilación de datos hasta la visualización de resultados, utilizando técnicas avanzadas de inteligencia artificial y procesamiento del lenguaje natural.

## Objetivos
- Replicar el análisis en tiempo real de discursos políticos, evaluando la recepción pública hacia los mensajes a través del análisis de comentarios en redes sociales.
- Aplicar el análisis de sentimientos a comentarios en directo en plataformas de streaming, adaptando el sistema para procesar eficazmente comentarios en español.
- Analizar la experiencia del cliente y la percepción de la marca en el entorno digital a través del análisis de comentarios y reseñas en YouTube y Reddit.
- Implementar paneles estáticos y dinámicos en Power BI para facilitar la visualización de resultados en tiempo real.

===

## Instalación y Dependencias

Para utilizar este proyecto, asegúrate de tener instaladas las siguientes librerías:

- Python
- PyTorch
- NumPy
- Requests
- Pandas
- Time
- Selenium
- BeautifulSoup4
- Re
- Sys
- PRAW
- googleapiclient.discovery
- IPython.display
- urllib.request
- Pickle
- String
- NLTK
- WordCloud
- Keras
- Collections
- Matplotlib
- Seaborn
- Warnings
- XGBoost
- Scikit-learn
- TensorFlow

Puedes instalar estas librerías utilizando `pip` de la siguiente manera:

```bash
pip install torch numpy requests pandas selenium beautifulsoup4 praw google-api-python-client ipython urllib3 nltk wordcloud keras matplotlib seaborn xgboost scikit-learn tensorflow
```

Asegúrate de tener un entorno de Python configurado correctamente y ejecutar el comando de instalación en tu terminal o entorno virtual para disponer de todas las dependencias necesarias para ejecutar el código de este proyecto.

## Datasets

| Dataset | Download |
| ---     | ---      |
| training.1600000.processed.noemoticon.csv.part1 | [Download](https://github.com/Javigrz/Codigo_TFG_Javier_Gil/blob/main/data/0%20train/kaggle/sentiment/training.1600000.processed.noemoticon.csv.part1) | 
| training.1600000.processed.noemoticon.csv.part2 | [Download](https://github.com/Javigrz/Codigo_TFG_Javier_Gil/blob/main/data/0%20train/kaggle/sentiment/training.1600000.processed.noemoticon.csv.part2) |
| training.1600000.processed.noemoticon.csv.part3 | [Download](https://github.com/Javigrz/Codigo_TFG_Javier_Gil/blob/main/data/0%20train/kaggle/sentiment/training.1600000.processed.noemoticon.csv.part3) |
| training.1600000.processed.noemoticon.csv.part4 | [Download](https://github.com/Javigrz/Codigo_TFG_Javier_Gil/blob/main/data/0%20train/kaggle/sentiment/training.1600000.processed.noemoticon.csv.part4) |
| training.1600000.processed.noemoticon.csv.part5 | [Download](https://github.com/Javigrz/Codigo_TFG_Javier_Gil/blob/main/data/0%20train/kaggle/sentiment/training.1600000.processed.noemoticon.csv.part5) |
| training.1600000.processed.noemoticon.csv.part6 | [Download](https://github.com/Javigrz/Codigo_TFG_Javier_Gil/blob/main/data/0%20train/kaggle/sentiment/training.1600000.processed.noemoticon.csv.part6) |
| training.1600000.processed.noemoticon.csv.part7 | [Download](https://github.com/Javigrz/Codigo_TFG_Javier_Gil/blob/main/data/0%20train/kaggle/sentiment/training.1600000.processed.noemoticon.csv.part7) |
| training.1600000.processed.noemoticon.csv.part8 | [Download](https://github.com/Javigrz/Codigo_TFG_Javier_Gil/blob/main/data/0%20train/kaggle/sentiment/training.1600000.processed.noemoticon.csv.part8) |
| training.1600000.processed.noemoticon.csv.part9 | [Download](https://github.com/Javigrz/Codigo_TFG_Javier_Gil/blob/main/data/0%20train/kaggle/sentiment/training.1600000.processed.noemoticon.csv.part9) |
| training.1600000.processed.noemoticon.csv.part10 | [Download](https://github.com/Javigrz/Codigo_TFG_Javier_Gil/blob/main/data/0%20train/kaggle/sentiment/training.1600000.processed.noemoticon.csv.part10) |
| training.1600000.processed.noemoticon.csv.part11 | [Download](https://github.com/Javigrz/Codigo_TFG_Javier_Gil/blob/main/data/0%20train/kaggle/sentiment/training.1600000.processed.noemoticon.csv.part11) |
| training.1600000.processed.noemoticon.csv.part12 | [Download](https://github.com/Javigrz/Codigo_TFG_Javier_Gil/blob/main/data/0%20train/kaggle/sentiment/training.1600000.processed.noemoticon.csv.part12) |
| pol_4chan_comments.csv | [Download](https://github.com/Javigrz/Codigo_TFG_Javier_Gil/blob/main/data/1%20Politics/4chan/pol_4chan_comments.csv) |
| yt_video_comments.csv (Anouncement) | [Download](https://github.com/Javigrz/Codigo_TFG_Javier_Gil/blob/main/data/1%20Politics/TrumpCampaign/Anouncement/yt_video_comments.csv) |
| yt_video_stats.csv (Anouncement) | [Download](https://github.com/Javigrz/Codigo_TFG_Javier_Gil/blob/main/data/1%20Politics/TrumpCampaign/Anouncement/yt_video_stats.csv) |
| yt_video_comments.csv (VictorySpeech) | [Download](https://github.com/Javigrz/Codigo_TFG_Javier_Gil/blob/main/data/1%20Politics/TrumpCampaign/VictorySpeech/yt_video_comments.csv) |
| yt_video_stats.csv (VictorySpeech) | [Download](https://github.com/Javigrz/Codigo_TFG_Javier_Gil/blob/main/data/1%20Politics/TrumpCampaign/VictorySpeech/yt_video_stats.csv) |
| reddit_coments.csv | [Download](https://github.com/Javigrz/Codigo_TFG_Javier_Gil/blob/main/data/1%20Politics/redditPolitics/reddit_coments.csv) |
| reddit_publications.csv | [Download](https://github.com/Javigrz/Codigo_TFG_Javier_Gil/blob/main/data/1%20Politics/redditPolitics/reddit_publications.csv) |

## Uso
- for train
  ```
  [3 model/pythonModel.ipynb](https://github.com/Javigrz/Codigo_TFG_Javier_Gil/blob/main/3%20model/pythonModel.ipynb)
  ```
- for test
  ```
  [4 prediction/generate_predictions_comments.ipynb](https://github.com/Javigrz/Codigo_TFG_Javier_Gil/blob/main/4%20prediction/generate_predictions_comments.ipynb)
  ```
## Pretrained model
| Model | Download |
| ---     | ---   |
| Sentiment Model | [download](https://github.com/Javigrz/Codigo_TFG_Javier_Gil/blob/main/3%20model/sentiment_model.h5) |
| tokenizer | [download](https://github.com/Javigrz/Codigo_TFG_Javier_Gil/blob/main/3%20model/tokenizer.pickle) |


## Estructura del Proyecto
```
|—— .DS_Store
|—— .gitignore
|—— 1 extract_Data
|    |—— 4chanAPI.ipynb
|    |—— InstagramScrapping.ipynb
|    |—— RedditAPI.ipynb
|    |—— YouTubeAPI.ipynb
|    |—— YouTubeLiveAPI.ipynb
|    |—— tiktokScrapping.ipynb
|—— 2 analysis
|    |—— train_data_analysis.ipynb
|—— 3 model
|    |—— keras
|        |—— checkpoint
|        |—— keras.data-00000-of-00001
|        |—— keras.index
|        |—— keras_weights.weights.h5
|    |—— pyspark
|        |—— pysparkML.ipynb
|    |—— pythonModel.ipynb
|    |—— sentiment_model.h5
|    |—— tokenizer.pickle
|—— 4 prediction
|    |—— generate_predictions_comments.ipynb
|—— 5 visuals
|    |—— RealTime
|        |—— YouTubeLiveAPI.py
|        |—— __pycache__
|            |—— YouTubeLiveAPI.cpython-311.pyc
|        |—— powerBIPanel.ipynb
|        |—— powerBIWeb.py
|    |—— predictedData
|—— config
|    |—— __pycache__
|        |—— config.cpython-311.pyc
|        |—— powerBI.cpython-311.pyc
|        |—— reddit.cpython-311.pyc
|        |—— youtube.cpython-311.pyc
|        |—— youtube.cpython-312.pyc
|    |—— powerBI.py
|    |—— reddit.py
|    |—— youtube.py
|—— data
|    |—— .DS_Store
|    |—— 0 train
|        |—— .DS_Store
|        |—— kaggle
|            |—— .DS_Store
|            |—— sentiment
|                |—— training.1600000.processed.noemoticon.csv.part1
|                |—— training.1600000.processed.noemoticon.csv.part10
|                |—— training.1600000.processed.noemoticon.csv.part11
|                |—— training.1600000.processed.noemoticon.csv.part12
|                |—— training.1600000.processed.noemoticon.csv.part2
|                |—— training.1600000.processed.noemoticon.csv.part3
|                |—— training.1600000.processed.noemoticon.csv.part4
|                |—— training.1600000.processed.noemoticon.csv.part5
|                |—— training.1600000.processed.noemoticon.csv.part6
|                |—— training.1600000.processed.noemoticon.csv.part7
|                |—— training.1600000.processed.noemoticon.csv.part8
|                |—— training.1600000.processed.noemoticon.csv.part9
|    |—— 1 Politics
|        |—— .DS_Store
|        |—— 4chan
|            |—— pol_4chan_comments.csv
|        |—— TrumpCampaign
|            |—— Anouncement
|                |—— yt_video_comments.csv
|                |—— yt_video_stats.csv
|            |—— VictorySpeech
|                |—— yt_video_comments.csv
|                |—— yt_video_stats.csv
|        |—— redditPolitics
|            |—— reddit_coments.csv
|            |—— reddit_publications.csv
|    |—— 2 Streaming
|        |—— .DS_Store
|        |—— Ibai
|            |—— yt_video_comments.csv
|            |—— yt_video_ids.txt
|            |—— yt_video_stats.csv
|    |—— 3 Client Experience
|        |—— Apple
|            |—— YT_LaMMordida
|                |—— yt_video_comments.csv
|                |—— yt_video_ids.txt
|                |—— yt_video_stats.csv
|            |—— apple_all_data.csv
|            |—— reddit
|                |—— reddit_coments.csv
|                |—— reddit_publications.csv
|    |—— 4 LiveYT
|        |—— liveComments.csv
|    |—— format_data_pbi.ipynb
```
## License
![alt text](logoUPM.png)