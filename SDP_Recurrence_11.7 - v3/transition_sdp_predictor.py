import json
from typing import List

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from overrides import overrides
from transition_sdp_reader import parse_sentence


@Predictor.register('transition_predictor_sdp')
class SDPParserPredictor(Predictor):
    """
    预测器模型
    """

    def predict(self, sentence: str) -> JsonDict:
        """
        根据CoNLL格式的句子数据，去预测该句子的语义依存图（SDG）。
        只需要预先完成分词处理即可，词性、语义关系都可以mask掉。
        :param sentence: CoNLL格式的句子数据。
        :return: 返回标注了语义依存关系的CoNLL格式句子数据。
        """
        return self.predict_json({"sentence": sentence,})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        将Json化后的句子数据转成实例，以便预测。
        :param json_dict:``{"sentence": "..."}``.
        :return:一个实例，包含了tokens和meta_info两个域
        """
        # ret = parse_sentence(json.dumps(json_dict))
        ret = parse_sentence(json_dict["sentence"])

        tokens = ret["tokens"]
        meta_info = ret["meta_info"]
        pos_tags = ret["pos_tags"]
        token_characters = ret["token_characters"]
        arc_indices = ret["arc_indices"]
        arc_tags = ret["arc_tags"]

        return self._dataset_reader.text_to_instance(tokens=tokens, 
                                                     meta_info=meta_info,
                                                     pos_tags=pos_tags,
                                                     token_characters=token_characters,
                                                     arc_indices=arc_indices,
                                                     arc_tags=arc_tags
                                                     )

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        """
        对实例化的单个数据进行预测
        :param instance:实例化后的CoNLL格式数据
        :return:预测结果（CoNLL格式）
        """
        outputs = self._model.forward_on_instance(instance)

        # ret_dict = sdp_trans_outputs_into_conll(outputs)

        return sanitize(outputs)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        """
        对实例化的批量数据进行预测
        :param instances:实例化后的批量CoNLL格式数据
        :return:批量预测结果（CoNLL格式）
        """
        outputs_batch = self._model.forward_on_instances(instances)

        ret_dict_batch = [[] for i in range(len(outputs_batch))]
        # for outputs_idx in range(len(outputs_batch)):
        #     try:
        #         ret_dict_batch[outputs_idx] = sdp_trans_outputs_into_conll(outputs_batch[outputs_idx])
        #     except:
        #         print(json.loads(outputs_batch[outputs_idx]["tokens"]))

        return sanitize(ret_dict_batch)


def sdp_trans_outputs_into_conll(outputs):
    """
    根据模型预测结果，输出CoNLL格式数据
    :param outputs:模型预测结果字典
    :return:CoNLL格式数据
    """
    edge_list = outputs["edge_list"]
    edge_list.sort(key=lambda x: x[0])
    tokens = outputs["tokens"]
    pos_tags = outputs["pos_tags"]

    conll_result = []
    for edge in edge_list:
        conll_result.append(
            "{:d}\t{:s}\t{:s}\t{:s}\t{:s}\t_\t{:d}\t{:s}\t_\t_\n".format(edge[0], tokens[edge[0]], tokens[edge[0]],
                                                                         pos_tag[edge[0]], pos_tag[edge[0]], edge[1],
                                                                         edge[2]))
    return conll_result