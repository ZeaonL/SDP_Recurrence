config = {
    'train_file_path' : 'Program/data/SemEval-2016-master/train/text.train.conll',
    'dev_file_path' : 'Program/data/SemEval-2016-master/test/text.test.conll',
    'test_file_path' : 'Program/data/SemEval-2016-master/test/text.test.conll',

    # 'dev_file_path' : 'Program/data/SemEval-2016-master/validation/text.valid.conll',

    'pretrain_path' : 'Program/data/giga/giga.100.txt',
    'pretrain_char_path' : 'Program/data/giga/giga.chars.100.txt',

    # 'train_file_path' : 'Program/SDP/microdata/trail.train.conll',
    # 'dev_file_path' : 'Program/SDP/microdata/trail.dev.conll',
    # 'test_file_path' : 'Program/SDP/microdata/trail.test.conll',

    # 'train_file_path' : 'Program/SDP/SDP_Recurrence_new/microdata/trail.train.conll',
    # 'dev_file_path' : 'Program/SDP/SDP_Recurrence_new/microdata/trail.dev.conll',
    # 'test_file_path' : 'Program/SDP/SDP_Recurrence_new/microdata/trail.test.conll',

    'batch_size' : 30,
    # 'shuffle' : True,
    # 'lr' : 2e-3,
    # 'device' : 'cuda:4',
    # 'min_freq' : 3,
    # 'batch_first' : True,
    'model_path' : 'Program/SDP/model.pt',
    'num_epochs' : 500,
    'word_dim' : 200,
    'hidden_dim' : 300,
    'action_dim' : 100,
    'num_layers' : 2,
    'recurrent_dropout_probability' : 0.2,
    'layer_dropout_probability' : 0.2, 
    'input_dropout' : 0.2,
}