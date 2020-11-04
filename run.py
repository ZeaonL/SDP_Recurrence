import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'

import tempfile
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader

import allennlp
from allennlp.data import allennlp_collate
from allennlp.data import DatasetReader, Instance
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, Embedding, Seq2SeqEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
# from allennlp.modules.seq2vec_encoders import StackedAlternatingLstmSeq2VecEncoder
from allennlp.training.trainer import Trainer, GradientDescentTrainer
from allennlp.training.optimizers import AdamOptimizer
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer, TokenCharactersIndexer
from allennlp.nn import InitializerApplicator, RegularizerApplicator

from config import config

from my_token_characters_encoder import TokenCharactersEncoder
from transition_sdp_reader import SDPDatasetReader
from transition_parser_sdp import TransitionParser
from transition_sdp_metric import MyMetric

def build_trainer(
    model: Model,
    serialization_dir: str,
    train_loader: DataLoader,
    dev_loader: DataLoader
) -> Trainer:
    parameters = [
        [n, p]
        for n, p in model.named_parameters() if p.requires_grad
    ]
    optimizer = AdamOptimizer(parameters)
    trainer = GradientDescentTrainer(
                model=model,
                serialization_dir=serialization_dir,
                data_loader=train_loader,
                validation_data_loader=dev_loader,
                num_epochs=config['num_epochs'],
                optimizer=optimizer
            )
    
    return trainer

def read_data(
    reader: DatasetReader
) -> Tuple[Iterable[Instance], Iterable[Instance]]:

    train_set = reader.read(config['train_file_path'])
    dev_set = reader.read(config['dev_file_path'])

    return train_set, dev_set

def build_dataset_reader() -> DatasetReader:

    token_indexers = {'tokens': SingleIdTokenIndexer(namespace='tokens', lowercase_tokens=True)}
    characters_indexers = {'token_characters': TokenCharactersIndexer(namespace='token_characters')}
    action_indexers = {'actions': SingleIdTokenIndexer(namespace='actions')}
    arc_tag_indexers = {'arc_tags': SingleIdTokenIndexer(namespace='arc_tags')}

    return SDPDatasetReader(token_indexers = token_indexers,
                            action_indexers = action_indexers,
                            arc_tag_indexers = arc_tag_indexers,
                            characters_indexers = characters_indexers)

def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    return Vocabulary.from_instances(instances)

def build_model(vocab: Vocabulary) -> Model:

    vocab_size = vocab.get_vocab_size("tokens")
    text_field_embedder = BasicTextFieldEmbedder(
        {"tokens": Embedding(embedding_dim=config['word_dim'], num_embeddings=vocab_size)})
    # encoder = StackedAlternatingLstmSeq2VecEncoder(input_size=100, 
    #                         hidden_size=400, 
    #                         num_layers=1, 
    #                         recurrent_dropout_probability=0.33,
    #                         use_highway=True)
    # pos_tagger_encoder = StackedAlternatingLstmSeq2VecEncoder(input_size=200, 
    #                                     hidden_size=300, 
    #                                     num_layers=2, 
    #                                     recurrent_dropout_probability=0.33,
    #                                     use_highway=True)
    metric = MyMetric()
    action_embedding = Embedding(vocab_namespace='actions', embedding_dim=config['action_dim'], num_embeddings=vocab.get_vocab_size('actions'))

    return TransitionParser(vocab=vocab, 
                            text_field_embedder=text_field_embedder, 
                            word_dim=config['word_dim'],
                            hidden_dim=config['hidden_dim'],
                            action_dim=config['action_dim'],
                            num_layers=config['num_layers'],
                            metric=metric,
                            recurrent_dropout_probability=config['recurrent_dropout_probability'],
                            layer_dropout_probability=config['layer_dropout_probability'],
                            same_dropout_mask_per_instance=True,
                            input_dropout=config['input_dropout'],
                            action_embedding=action_embedding,
                            #  initializer=,
                            #  regularizer=
                            ).to('cuda')

def build_data_loaders(
    train_data: torch.utils.data.Dataset,
    dev_data: torch.utils.data.Dataset,
) -> Tuple[allennlp.data.DataLoader, allennlp.data.DataLoader]:

    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, collate_fn=allennlp_collate)
    dev_loader = DataLoader(dev_data, batch_size=config['batch_size'], shuffle=True, collate_fn=allennlp_collate)
    return train_loader, dev_loader

def run_training_loop():
    reader = build_dataset_reader()
    train_set, dev_set = read_data(reader)

    vocab = build_vocab(train_set + dev_set)
    # TokenCharactersEncoder.from_params()

    model = build_model(vocab)

    train_set.index_with(vocab)
    dev_set.index_with(vocab)

    train_loader, dev_loader = build_data_loaders(train_set, dev_set)

    # k = next(iter(train_loader))
    # print(next(iter(train_loader))['tokens']['tokens']['tokens'])

    with tempfile.TemporaryDirectory() as serialization_dir:
        trainer = build_trainer(
            model,
            serialization_dir,
            train_loader,
            dev_loader
        )
        trainer.train()     # 先进入train的forward，紧接着再进入metric，再紧接着进入了dev的forward

    return model, reader

model, dataset_reader = run_training_loop()
vocab = model.vocab