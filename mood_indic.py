import gensim
import gzip
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

mood_file = "mood_data.txt"

def add_data(text):
  file = open(mood_file, "a")
  file.write(text)
  file.close()

def read_data(file_name):
  file = open(file_name, "r")
  lines = file.readlines()
  file.close()

  for i in range(len(lines)):
    lines[i] = lines[i][:-1]

  return lines

def remove_stop_words(words):
  return [i for i in words if i not in stop_words]

def process_lines(lines):
  processed_lines = []
  for l in lines:
    if "of" in l:
      hi = 9
    l = ' '.join(remove_stop_words(l.split(' ')))
    processed_lines.append(gensim.utils.simple_preprocess(l))

  return processed_lines

def train_model():
  data = process_lines(read_data(mood_file))
  model = gensim.models.Word2Vec (data, vector_size=50, window=10, min_count=2, workers=10)
  model.train(data, total_examples=len(data), epochs=10)
  return model

stop_words = read_data("stopwords.txt")
model = train_model()

w = "done"
vals = model.wv.most_similar (positive=w,topn=15)
for v in vals:
  print(v)
