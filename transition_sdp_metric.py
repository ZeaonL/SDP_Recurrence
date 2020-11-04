import torch

class MyMetric():

    def __init__(self, eps=1e-8):
        super().__init__()

        self.eps = eps

        self.n_predict = 0.0
        self.n_total = 0.0
        self.correct_arcs = 0.0
        self.correct_rels = 0.0

    def __repr__(self):
        s = f"LF: {self.LF}, UF: {self.UF} "
        # s += f"UAS: {self.uas:6.2%} LAS: {self.las:6.2%}"
        return s

    def __call__(self, preds, golds):
        '''
        preds: [[(end, head, 'rel'), ...],
                [(end, head, 'rel'), ...],
                [(end, head, 'rel'), ...]]
        golds: [{'arc_indices': [(end, head), ...], 
                 'arc_tags': ['rel', ...]}, 
                {'arc_indices': [(end, head), ...], 
                 'arc_tags': ['rel', ...]}, 
                {'arc_indices': [(end, head), ...], 
                 'arc_tags': ['rel', ...]}]
        '''
        self.n_predict += sum([len(sent) for sent in preds])
        self.n_total += sum([len(sent['arc_indices']) for sent in golds])
        for i, sent in enumerate(preds):
            for arc in sent:
                
                try: 
                    k = golds[i]['arc_indices'].index(arc[:2])    # NOTE 还不知道这样写对不对
                    self.correct_arcs += 1
                    if golds[i]['arc_tags'][k] == arc[2]:
                        self.correct_rels += 1
                except ValueError:
                    pass

    def reset(self):
        self.n_predict = 0.0
        self.n_total = 0.0
        self.correct_arcs = 0.0
        self.correct_rels = 0.0

    @property
    def LF_Precision(self):
        return self.correct_rels / (self.n_predict + self.eps)
    
    @property
    def LF_Recall(self):
        return self.correct_rels / (self.n_total + self.eps)

    @property
    def UF_Precision(self):
        return self.correct_arcs / (self.n_predict + self.eps)

    @property
    def UF_Recall(self):
        return self.correct_arcs / (self.n_total + self.eps)

    @property
    def LF(self):
        return (2*self.LF_Recall*self.LF_Precision) / (self.LF_Recall + self.LF_Precision + self.eps)

    @property
    def UF(self):
        return (2*self.UF_Recall*self.UF_Precision) / (self.UF_Recall + self.UF_Precision + self.eps)