import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'

import tempfile
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader

import allennlp
from allennlp.data import allennlp_collate
# from allennlp.common import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data import Vocabulary
# from allennlp.data.fields import LabelField, TextField
# from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, Embedding, Seq2SeqEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import StackedAlternatingLstmSeq2VecEncoder
# from allennlp.nn import util
# from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import Trainer, GradientDescentTrainer
from allennlp.training.optimizers import AdamOptimizer
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer, TokenCharactersIndexer
from allennlp.training.metrics import Metric
from allennlp.nn import InitializerApplicator, RegularizerApplicator

from config import config

from transition_sdp_reader import SDPDatasetReader
from transition_parser_sdp import TransitionParser

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
                num_epochs=50,
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
                            arc_tag_indexers = arc_tag_indexers)

def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    return Vocabulary.from_instances(instances)

def build_model(vocab: Vocabulary) -> Model:

    vocab_size = vocab.get_vocab_size("tokens")
    text_field_embedder = BasicTextFieldEmbedder(
        {"tokens": Embedding(embedding_dim=100, num_embeddings=vocab_size)})
    encoder = StackedAlternatingLstmSeq2VecEncoder(input_size=100, 
                            hidden_size=400, 
                            num_layers=1, 
                            recurrent_dropout_probability=0.33,
                            use_highway=True)
    pos_tagger_encoder = StackedAlternatingLstmSeq2VecEncoder(input_size=200, 
                                        hidden_size=300, 
                                        num_layers=2, 
                                        recurrent_dropout_probability=0.33,
                                        use_highway=True)
    metric = Metric()       # TODO 存疑
    action_embedding = Embedding(vocab_namespace='actions', embedding_dim=50, num_embeddings=vocab.get_vocab_size('actions'))

    # TODO 这里应该还缺控制器，但是我还不清楚，在调试过程中慢慢添加
    return TransitionParser(vocab=vocab, 
                            text_field_embedder=text_field_embedder, 
                            word_dim=200,
                            hidden_dim=200,
                            action_dim=50,
                            num_layers=2,
                            mces_metric=metric,
                            recurrent_dropout_probability=0.2,
                            layer_dropout_probability=0.2,
                            same_dropout_mask_per_instance=True,
                            input_dropout=0.2,
                            #  pos_tag_embedding=,
                            action_embedding=action_embedding,
                            pos_tagger_encoder=pos_tagger_encoder
                            #  initializer=,
                            #  regularizer=
                            ).to('cuda')

def build_data_loaders(
    train_data: torch.utils.data.Dataset,
    dev_data: torch.utils.data.Dataset,
) -> Tuple[allennlp.data.DataLoader, allennlp.data.DataLoader]:

    train_loader = DataLoader(train_data, batch_size=30, shuffle=True, collate_fn=allennlp_collate)
    dev_loader = DataLoader(dev_data, batch_size=30, shuffle=True, collate_fn=allennlp_collate)
    return train_loader, dev_loader

def run_training_loop():
    reader = build_dataset_reader()
    train_set, dev_set = read_data(reader)

    vocab = build_vocab(train_set + dev_set)
    model = build_model(vocab)

    train_set.index_with(vocab)
    dev_set.index_with(vocab)

    train_loader, dev_loader = build_data_loaders(train_set, dev_set)

    # TODO 这里的字典格式非常冗杂，为什么会有3个嵌套的字典，在代码上进行修改的复杂程度还是未知数，看看模型部分怎么调用的。
    # print(next(iter(train_loader))['tokens']['tokens']['tokens'])

    with tempfile.TemporaryDirectory() as serialization_dir:
        trainer = build_trainer(
            model,
            serialization_dir,
            train_loader,
            dev_loader
        )
        trainer.train()

    return model, dataset_reader

model, dataset_reader = run_training_loop()
vocab = model.vocab

print()

# from allennlp.commands.train import train_model_from_file