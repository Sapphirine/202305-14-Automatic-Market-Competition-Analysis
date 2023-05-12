from flask import Flask, request
from datetime import datetime
from flask_cors import CORS
import json
import time
app = Flask(__name__)
CORS(app)  

from steam_data import SteamData
sd = SteamData()

@app.route('/submit', methods=['POST']) 
def submit():
  category = request.form.get('category')
  tag = request.form.get('tag')
  genre = request.form.get('genre')
  timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
  print(f'\n{timestamp} | Category: {category} | Tag: {tag} | Genre: {genre}')
  if not sd.user_input(category, tag, genre):
    return 'No game found, please try again.'
  # time.sleep(0.5)
  return 'Success'

@app.route('/data_process', methods=['POST'])
def data_process():
  sd.drop_short_articles()
  sd.train_test_split()
  # time.sleep(0.5)
  print('Data processing finished.')
  return 'Success' 

@app.route('/train_lda', methods=['POST'])
def train_lda():
  sd.train_lda()
  # time.sleep(0.5)

  print('LDA training finished.')
  return 'Success'

@app.route('/similar_docs', methods=['POST'])
def similar_docs():
  sd.get_most_similar_documents()
  print('Similar documents finished.')
  return json.dumps({'game_names': sd.most_similar_game_names.tolist(), 'game_desc': sd.most_similar_game_descriptions.tolist()})

@app.route('/game_stats', methods=['POST']) 
def game_stats():
  sd.game_statistics_analysis()
  print('Game statistics finished.')
  return json.dumps([
    'result/game_statistics_analysis_1.png',
    'result/game_statistics_analysis_2.png',
    'result/game_statistics_analysis_3.png',
    'result/game_statistics_analysis_4.png',
  ])

@app.route('/pos_user_reviews', methods=['POST'])
def pos_user_reviews(): 
  cluster_summary = sd.select_reviews(1)
  print('Pos reviews finished.')
  print(cluster_summary)
  return json.dumps({'cluster_summary': cluster_summary, 'word_cloud': ['result/score1_cloud1.png', 'result/score1_cloud2.png']})

@app.route('/neg_user_reviews', methods=['POST'])
def neg_user_reviews(): 
  cluster_summary = sd.select_reviews(0)
  print('Neg reviews finished.')
  print(cluster_summary)
  return json.dumps({'cluster_summary': cluster_summary, 'word_cloud': ['result/score0_cloud1.png', 'result/score0_cloud2.png']})

if __name__ == '__main__': 
  app.run(host='0.0.0.0', port=5000) 