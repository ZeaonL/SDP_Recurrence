import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'

from config import config

from transition_parser_sdp import TransitionParser
from transition_sdp_predictor import SDPParserPredictor

def load_model_loop(sentence, predictor):
    # print(sentence)
    predict = predictor.predict(sentence)
    # print(predict)

model = TransitionParser.load(config['model_path'])
model.metric.reset()

vocab = model.vocab
reader = model.reader
predictor = SDPParserPredictor(model, reader)

fp = open(config['test_file_path'], 'r', encoding='utf-8')
conlls = fp.read().split('\n\n')[:-1]
fp.close()
for sentence in conlls:
    load_model_loop(sentence, predictor)
print(model.metric)
