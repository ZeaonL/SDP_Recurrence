from allennlp.training.metrics.metric import Metric
from overrides import overrides

@Metric.register("MyMetric")
class MyMetric(Metric):
    """
    计算评价指标
    """
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

    @overrides
    def __call__(self, preds, golds, masks=None):
        """
        :param preds: 预测结果
                [[(end, head, 'rel'), ...],
                [(end, head, 'rel'), ...],
                [(end, head, 'rel'), ...]]
        :param golds: 标准结果
        :param masks: padding的mask标记
        """
        self.n_predict += sum([len(sent) for sent in preds])
        self.n_total += sum([len(sent['arc_indices']) for sent in golds])
        for i, sent in enumerate(preds):
            for arc in sent:
                try: 
                    k = golds[i]['arc_indices'].index(arc[:2])
                    self.correct_arcs += 1
                    if golds[i]['arc_tags'][k] == arc[2]:
                        self.correct_rels += 1
                except ValueError:
                    pass

    @overrides
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