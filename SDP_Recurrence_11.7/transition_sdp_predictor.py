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
        return self.predict_json({"sentence": sentence})

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

        return self._dataset_reader.text_to_instance(tokens=tokens, meta_info=[meta_info])

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        """
        对实例化的单个数据进行预测
        :param instance:实例化后的CoNLL格式数据
        :return:预测结果（CoNLL格式）
        """
        outputs = self._model.forward_on_instance(instance)

        ret_dict = sdp_trans_outputs_into_conll(outputs)

        return sanitize(ret_dict)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        """
        对实例化的批量数据进行预测
        :param instances:实例化后的批量CoNLL格式数据
        :return:批量预测结果（CoNLL格式）
        """
        outputs_batch = self._model.forward_on_instances(instances)

        ret_dict_batch = [[] for i in range(len(outputs_batch))]
        for outputs_idx in range(len(outputs_batch)):
            try:
                ret_dict_batch[outputs_idx] = sdp_trans_outputs_into_conll(outputs_batch[outputs_idx])
            except:
                print(json.loads(outputs_batch[outputs_idx]["tokens"]))

        return sanitize(ret_dict_batch)


# TODO
def sdp_trans_outputs_into_conll(outputs):
    edge_list = outputs["edge_list"]
    tokens = outputs["tokens"]
    meta_info = outputs["meta_info"]
    tokens_range = outputs["tokens_range"]
    frame = outputs["frame"]
    pos_tag = outputs["pos_tag"]
    node_label = outputs["node_label"]

    meta_info_dict = json.loads(meta_info)
    ret_dict = {}
    for key in ["id", "flavor", "version", "framework", "time", "input"]:
        ret_dict[key] = meta_info_dict[key]

    ###Node Labels /Properties /Anchoring
    nodes_info_list = []
    is_node_has_edge = [False for i in range(len(tokens))]
    for edge_info in edge_list:
        is_node_has_edge[edge_info[1] - 1] = True
        is_node_has_edge[edge_info[0] - 1] = True

    for nodes_info_idx in range(len(tokens)):
        if is_node_has_edge[nodes_info_idx] == False or node_label[nodes_info_idx] == 'non':
            continue

        node_info_dict = {"anchors": [{"from": tokens_range[nodes_info_idx][0], "to": tokens_range[nodes_info_idx][1]}], \
                          "id": nodes_info_idx,
                          "label": node_label[nodes_info_idx],
                          "properties": [],
                          "values": []}

        if pos_tag[nodes_info_idx] != 'non':
            node_info_dict["properties"].append("pos")
            node_info_dict["values"].append(pos_tag[nodes_info_idx])

        if frame[nodes_info_idx] != 'non':
            node_info_dict["properties"].append("frame")
            node_info_dict["values"].append(frame[nodes_info_idx])

        if len(node_info_dict["properties"]) == 0:
            node_info_dict.pop('properties')
            node_info_dict.pop('values')

        nodes_info_list.append(node_info_dict)

    ret_dict["nodes"] = nodes_info_list

    ###Directed Edges /Edge Labels /Edge Properties
    # 1. need post-processing the mutil-label edge
    # 2. edge prorperty, i.e remote-edges
    edges_info = []
    for edge_info in edge_list:
        if (edge_info[-1] == 'ROOT' and edge_info[1] == 0) or node_label[edge_info[1] - 1] == 'non' or node_label[
            edge_info[0] - 1] == 'non':
            continue
        edges_info.append({"label": edge_info[-1],
                           "source": edge_info[1] - 1,
                           "target": edge_info[0] - 1,
                           })
    ret_dict["edges"] = edges_info

    ###Top Nodes
    top_node_list = []

    for edge_info in edge_list:
        if edge_info[-1] == 'ROOT' and edge_info[1] == 0 and node_label[edge_info[0] - 1] != 'non':
            top_node_list.append(edge_info[0] - 1)

    ret_dict["tops"] = top_node_list

    return ret_dict
