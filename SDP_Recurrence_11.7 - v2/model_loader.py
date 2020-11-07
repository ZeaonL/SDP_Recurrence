import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'

import tempfile
from typing import Dict, Iterable, List, Tuple
from allennlp.modules.token_embedders import TokenCharactersEncoder
import torch
from torch.utils.data import DataLoader

import allennlp
from allennlp.data import allennlp_collate
from allennlp.data import DatasetReader, Instance
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, Embedding, Seq2SeqEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import StackedAlternatingLstmSeq2VecEncoder, BagOfEmbeddingsEncoder
from allennlp.training.trainer import Trainer, GradientDescentTrainer
from allennlp.training.optimizers import AdamOptimizer
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer, TokenCharactersIndexer
from allennlp.nn import InitializerApplicator, RegularizerApplicator

from config import config

from transition_sdp_reader import SDPDatasetReader
from transition_parser_sdp import TransitionParser
from transition_sdp_metric import MyMetric

def read_data(
    reader: DatasetReader
) -> Tuple[Iterable[Instance], Iterable[Instance]]:
    
    test_set = reader.read(config['test_file_path'])

    return test_set

def build_dataset_reader() -> DatasetReader:

    token_indexers = {'tokens': SingleIdTokenIndexer(namespace='tokens', lowercase_tokens=True)}
    characters_indexers = {'token_characters': TokenCharactersIndexer(namespace='token_characters')}
    action_indexers = {'actions': SingleIdTokenIndexer(namespace='actions')}
    arc_tag_indexers = {'arc_tags': SingleIdTokenIndexer(namespace='arc_tags')}

    return SDPDatasetReader(token_indexers=token_indexers,
                            action_indexers=action_indexers,
                            arc_tag_indexers=arc_tag_indexers,
                            characters_indexers=characters_indexers
                            )

def build_data_loaders(
    test_data: torch.utils.data.Dataset
) -> Tuple[allennlp.data.DataLoader, allennlp.data.DataLoader]:

    test_loader = DataLoader(test_data, batch_size=30, shuffle=True, collate_fn=allennlp_collate)
    return test_loader

def load_model_loop(vocab, reader, model):

    test_set = read_data(reader)
    test_set.index_with(vocab)
    test_loader = build_data_loaders(test_set)

    # temp = next(iter(test_loader))
    # print(next(iter(test_loader)))

    for batch in test_loader:
        out_dict = model(tokens=batch['tokens'],
                         token_characters=batch['token_characters'],
                         metadata=batch['metadata'],
                         gold_actions=batch['gold_actions'],
                         arc_tags=batch['arc_tags'],
                         pos_tags=batch['pos_tags'])
        print(out_dict)


model = TransitionParser.load(config['model_path'])
vocab = model.vocab
reader = model.reader
load_model_loop(vocab, reader, model)
