import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '4'

import tempfile, copy
from typing import Dict, Iterable, List, Tuple
from allennlp.modules.token_embedders import TokenCharactersEncoder
import torch
from torch.utils.data import DataLoader

import allennlp
from allennlp.data import allennlp_collate
from allennlp.data.dataset_readers.dataset_reader import AllennlpDataset
# from allennlp.common import JsonDictm
from allennlp.data import DatasetReader, Instance
from allennlp.data import Vocabulary
# from allennlp.data.fields import LabelField, TextField
# from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, Embedding, Seq2SeqEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import StackedAlternatingLstmSeq2VecEncoder, BagOfEmbeddingsEncoder
# from allennlp.nn import util
# from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import Trainer, GradientDescentTrainer
from allennlp.training.optimizers import AdamOptimizer
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer, TokenCharactersIndexer
from allennlp.nn import InitializerApplicator, RegularizerApplicator

from config import config

from transition_sdp_reader import SDPDatasetReader
from transition_parser_sdp import TransitionParser
from transition_sdp_metric import MyMetric


def build_trainer(
        model: Model,
        serialization_dir: str,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        test_loader: DataLoader
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
        num_epochs=500,
        optimizer=optimizer
    )

    return trainer


def read_data(
        reader: DatasetReader
) -> Tuple[AllennlpDataset, AllennlpDataset, AllennlpDataset]:
    train_set = reader.read(config['train_file_path'])
    dev_set = reader.read(config['dev_file_path'])
    test_set = reader.read(config['test_file_path'])

    return train_set, dev_set, test_set


def build_dataset_reader() -> DatasetReader:
    token_indexers = {'tokens': SingleIdTokenIndexer(namespace='tokens', lowercase_tokens=True)}
    characters_indexers = {'token_characters': TokenCharactersIndexer(namespace='token_characters')}
    action_indexers = {'actions': SingleIdTokenIndexer(namespace='actions')}
    arc_tag_indexers = {'arc_tag': SingleIdTokenIndexer(namespace='arc_tag')}
    pos_tag_indexers = {'pos_tag': SingleIdTokenIndexer(namespace='pos_tag')}

    return SDPDatasetReader(token_indexers=token_indexers,
                            action_indexers=action_indexers,
                            arc_tag_indexers=arc_tag_indexers,
                            characters_indexers=characters_indexers,
                            pos_tag_indexers=pos_tag_indexers
                            )

def build_vocab(train_instances: AllennlpDataset,
                dev_instances: AllennlpDataset,
                test_instances: AllennlpDataset) -> Vocabulary:
    instances = train_instances + dev_instances + test_instances
    vocab = Vocabulary.from_instances(instances, pretrained_files={'tokens':config['pretrain_path'], 'token_characters':config['pretrain_char_path']},)
    return vocab


def build_model(vocab: Vocabulary, reader: DatasetReader) -> Model:
    pos_vocab_size = vocab.get_vocab_size("pos_tag")
    text_field_embedder = BasicTextFieldEmbedder(
        {"tokens": Embedding(embedding_dim=100,
                             pretrained_file=config['pretrain_path'],
                             vocab=vocab)})
    char_field_embedder = BasicTextFieldEmbedder(
        {"token_characters": TokenCharactersEncoder(Embedding(embedding_dim=100,
                                                              pretrained_file=config['pretrain_char_path'],
                                                              vocab=vocab),
                                                    BagOfEmbeddingsEncoder(100, True), 0.33)})
    pos_field_embedder = BasicTextFieldEmbedder(
        {"pos_tag": Embedding(embedding_dim=50, num_embeddings=pos_vocab_size)})

    metric = MyMetric()
    action_embedding = Embedding(vocab_namespace='actions', embedding_dim=50, num_embeddings=vocab.get_vocab_size('actions'))

    return TransitionParser(vocab=vocab,
                            reader=reader,
                            text_field_embedder=text_field_embedder,
                            char_field_embedder=char_field_embedder,
                            pos_tag_field_embedder=pos_field_embedder,
                            word_dim=250,
                            hidden_dim=200,
                            action_dim=50,
                            num_layers=2,
                            metric=metric,
                            recurrent_dropout_probability=0.2,
                            layer_dropout_probability=0.2,
                            same_dropout_mask_per_instance=True,
                            input_dropout=0.2,
                            action_embedding=action_embedding,
                            start_time=time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
                            )


def build_data_loaders(
        train_data: Iterable[Instance],
        dev_data: Iterable[Instance],
        test_data: Iterable[Instance]
) -> Tuple[allennlp.data.DataLoader, allennlp.data.DataLoader, allennlp.data.DataLoader]:
    train_loader = DataLoader(train_data, batch_size=30, shuffle=True, collate_fn=allennlp_collate)
    dev_loader = DataLoader(dev_data, batch_size=30, shuffle=True, collate_fn=allennlp_collate)
    test_loader = DataLoader(test_data, batch_size=30, shuffle=True, collate_fn=allennlp_collate)
    return train_loader, dev_loader, test_loader


def run_training_loop():
    reader = build_dataset_reader()
    train_set, dev_set, test_set = read_data(reader)

    vocab = build_vocab(train_set, dev_set, test_set)
    model = build_model(vocab, reader)

    train_set.index_with(vocab)
    dev_set.index_with(vocab)
    test_set.index_with(vocab)

    train_loader, dev_loader, test_loader = build_data_loaders(train_set, dev_set, test_set)

    # temp = next(iter(train_loader))
    # print(next(iter(train_loader)))

    with tempfile.TemporaryDirectory() as serialization_dir:
        trainer = build_trainer(
            model,
            serialization_dir,
            train_loader,
            dev_loader,
            test_loader
        )
        trainer.train()

    return model, reader


model, dataset_reader = run_training_loop()
vocab = model.vocab

