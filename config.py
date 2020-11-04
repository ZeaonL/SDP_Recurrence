config = {
    # 'train_file_path' : 'Program/data/SemEval-2016-master/train/text.train.conll',
    # 'dev_file_path' : 'Program/data/SemEval-2016-master/validation/text.valid.conll',
    'test_file_path' : 'Program/data/SemEval-2016-master/test/text.test.conll',
    # 'glove_file_path' : 'Program/data/glove/glove.6B.100d.txt',
    'train_file_path' : 'Program/SDP/microdata/trail.train.conll',
    'dev_file_path' : 'Program/SDP/microdata/trail.dev.conll',
    # 'test_file_path' : 'Program/SDP/microdata/trail.test.conll',
    # 'glove_file_path' : 'Program/SDP/microdata/trail.emb',
    'batch_size' : 30,
    'shuffle' : True,
    'lr' : 2e-3,
    'device' : 'cuda:4',
    'min_freq' : 3,
    'batch_first' : True,
    'out_file_path' : 'Program/SDP/output.txt',
    'num_epochs' : 500,
    'word_dim' : 200,
    'hidden_dim' : 300,
    'action_dim' : 100,
    'num_layers' : 2,
    'recurrent_dropout_probability' : 0.2,
    'layer_dropout_probability' : 0.2, 
    'input_dropout' : 0.2,
}