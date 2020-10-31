config = {
    # 'train_file_path' : 'Program/data/SemEval-2016-master/train/text.train.conll',
    # 'dev_file_path' : 'Program/data/SemEval-2016-master/validation/text.valid.conll',
    # 'test_file_path' : 'Program/data/SemEval-2016-master/test/text.test.conll',
    # 'glove_file_path' : 'Program/data/glove/glove.6B.100d.txt',
    'train_file_path' : 'Program/SDP/microdata/trail.train.conll',
    'dev_file_path' : 'Program/SDP/microdata/trail.dev.conll',
    'test_file_path' : 'Program/SDP/microdata/trail.test.conll',
    'glove_file_path' : 'Program/SDP/microdata/trail.emb',
    'batch_size' : 5000,
    'shuffle' : True,
    'lr' : 2e-3,
    'device' : 'cuda:4',
    'min_freq' : 3,
    'batch_first' : True,
    'out_file_path' : 'Program/SDP/output.txt'
}