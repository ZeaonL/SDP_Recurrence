import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

from typing import Dict, Tuple, List

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from overrides import overrides

class Relation(object):
    type = None

    def __init__(self, node, rel, remote=False):
        self.node = node
        self.rel = rel

    def show(self):
        print("Node:{},Rel:{} || ".format(self.node, self.rel), )


class Head(Relation): type = 'HEAD'


class Child(Relation): type = 'CHILD'


class Node(object):
    def __init__(self, info):
        self.id = info[0]

        self.label = info[1]
        self.pos_tag = info[3]
        self.heads, self.childs = [], []
        self.head_ids, self.child_ids = [], []

    def add_head(self, edge):
        assert edge[0] == self.id
        self.heads.append(Head(edge[6], edge[7]))
        self.head_ids.append(edge[6])

    def add_child(self, edge):
        assert edge[6] == self.id
        self.childs.append(Child(edge[0], edge[7]))
        self.child_ids.append(edge[0])

class Graph(object):
    def __init__(self, conll):
        conll = list(map(lambda x: x.split('\t'), conll.split('\n')))

        self.nodes = [Node([0, '<bos>', '_', '<bos>'])]
        count = 1
        for i, node in enumerate(conll):
            node[0], node[6] = int(node[0]), int(node[6])
            if node[0] == i + count:
                self.nodes.append(Node(node))
            else:
                count -= 1
        # print(self.nodes[0].label)

        self.edges = {}
        # conll : [1, '我们', '我们', 'PN', 'PN', '_', 3, 'Poss', '_', '_']
        for edge in conll:
            self.nodes[edge[6]].add_child(edge)
            self.nodes[edge[0]].add_head(edge)

        self.meta_info = conll

    def extract_token_info_from_data(self):
        tokens = [x.label for x in self.nodes]
        chars = []
        pos_tags = [x.pos_tag for x in self.nodes]
        return {"tokens": tokens,
                "pos_tags": pos_tags}

    def get_childs(self, id):
        childs = self.nodes[id].childs
        child_ids = [c.node for c in childs]
        return childs, child_ids

    def get_arc_info(self):
        tokens, arc_indices, arc_tags = [], [], []

        token_info = self.extract_token_info_from_data()

        ###Step 1: Extract surface token
        tokens = token_info["tokens"]
        pos_tags = token_info["pos_tags"]

        ###Step 2: Add arc label
        for node in self.nodes:
            for child in node.childs:
                arc_indices.append((child.node, node.id))        # from right to left
                arc_tags.append(child.rel)

        ret = {"tokens": tokens,
               "arc_indices": arc_indices,
               "arc_tags": arc_tags,
               "meta_info": self.meta_info,
               "pos_tag": pos_tags}

        return ret

def parse_sentence(sentence_blob: str):
    graph = Graph(sentence_blob)
    ret = graph.get_arc_info()
    return ret

def lazy_parse(text: str):
    conlls = text.split('\n\n')[:-1]
    for conll in conlls:
        yield parse_sentence(conll)

@DatasetReader.register("sdp_reader")
class SDPDatasetReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 action_indexers: Dict[str, TokenIndexer] = None,
                 arc_tag_indexers: Dict[str, TokenIndexer] = None,
                 characters_indexers : Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

        self._action_indexers = None
        # if action_indexers is not None and len(action_indexers) > 0:
        if action_indexers is not None:
            self._action_indexers = action_indexers
        self._arc_tag_indexers = None
        # if arc_tag_indexers is not None and len(arc_tag_indexers) > 0:
        if arc_tag_indexers is not None:
            self._arc_tag_indexers = arc_tag_indexers
        

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r', encoding='utf-8') as fp:
            logger.info("Reading SDP instances from conll dataset at: %s", file_path)
            for ret in lazy_parse(fp.read()):

                tokens = ret["tokens"]
                arc_indices = ret["arc_indices"]
                arc_tags = ret["arc_tags"]

                meta_info = ret["meta_info"]
                pos_tag = ret["pos_tag"]

                #In CoNLL2019, gold actions is not avaiable in test set.
                gold_actions = get_oracle_actions(tokens[1:], arc_indices, arc_tags) if arc_indices else None

                if gold_actions and gold_actions[-1] == '-E-':
                    print('-E-')
                    continue

                yield self.text_to_instance(tokens, arc_indices, arc_tags, gold_actions,
                                            [meta_info],
                                            pos_tag)

    @overrides
    def text_to_instance(self,  # type: ignore
                         tokens: List[str],
                         arc_indices: List[Tuple[int, int]] = None,
                         arc_tags: List[str] = None,
                         gold_actions: List[str] = None,
                         meta_info: List[str] = None,
                         pos_tag: List[str] = None) -> Instance:

        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        token_field = TextField([Token(t) for t in tokens], self._token_indexers)
        fields["tokens"] = token_field
        meta_dict = {"tokens": tokens}

        if pos_tag is not None:
            fields["pos_tag"] = SequenceLabelField(pos_tag, token_field, label_namespace="pos_tag")

        if arc_indices is not None and arc_tags is not None:
            meta_dict["arc_indices"] = arc_indices
            meta_dict["arc_tags"] = arc_tags
            fields["arc_tags"] = TextField([Token(a) for a in arc_tags], self._arc_tag_indexers)

        if gold_actions is not None:
            meta_dict["gold_actions"] = gold_actions
            fields["gold_actions"] = TextField([Token(a) for a in gold_actions], self._action_indexers)

        if meta_info is not None:
            meta_dict["meta_info"] = meta_info[0]

        fields["metadata"] = MetadataField(meta_dict)
        return Instance(fields)


def get_oracle_actions(annotated_sentence, directed_arc_indices, arc_tags):
    graph = {}
    for token_idx in range(len(annotated_sentence) + 1):
        graph[token_idx] = []

    # construct graph given directed_arc_indices and arc_tags
    # key: id_of_point
    # value: a list of tuples -> [(id_of_head1, label),(id_of_head2, label)，...]
    for arc, arc_tag in zip(directed_arc_indices, arc_tags):
        graph[arc[0]].append((arc[1], arc_tag))

    N = len(graph)  # N-1 point, 1 root point

    # i:head_point j:child_point
    top_down_graph = [[] for i in range(N)]  # N-1 real point, 1 root point => N point

    # i:child_point j:head_point ->Bool
    # partial graph during construction
    sub_graph = [[False for i in range(N)] for j in range(N)]

    for i in range(N):
        for head_tuple_of_point_i in graph[i]:
            head = head_tuple_of_point_i[0]
            top_down_graph[head].append(i)

    actions = []
    stack = [0]
    buffer = []
    deque = []

    for i in range(N - 1, 0, -1):
        buffer.append(i)

    # return if w1 is one head of w0
    def has_head(w0, w1):
        if w0 <= 0:
            return False
        for w in graph[w0]:
            if w[0] == w1:
                return True
        return False

    def has_unfound_child(w):
        for child in top_down_graph[w]:
            if not sub_graph[child][w]:
                return True
        return False

    # return if w has other head except the present one
    def has_other_head(w):
        if w <= 0:
            return False
        head_num = 0
        for h in sub_graph[w]:
            if h:
                head_num += 1
        if head_num + 1 < len(graph[w]):
            return True
        return False

    # return if w has any unfound head
    def lack_head(w):
        if w <= 0:
            return False
        head_num = 0
        for h in sub_graph[w]:
            if h:
                head_num += 1
        if head_num < len(graph[w]):
            return True
        return False

    # return if w has any unfound child in stack sigma
    # !!! except the top in stack
    def has_other_child_in_stack(stack, w):
        if w <= 0:
            return False
        for c in top_down_graph[w]:
            if c in stack \
                    and c != stack[-1] \
                    and not sub_graph[c][w]:
                return True
        return False

    # return if w has any unfound head in stack sigma
    # !!! except the top in stack
    def has_other_head_in_stack(stack, w):
        if w <= 0:
            return False
        for h in graph[w]:
            if h[0] in stack \
                    and h[0] != stack[-1] \
                    and not sub_graph[w][h[0]]:
                return True
        return False

    # return the relation between child: w0, head: w1
    def get_arc_label(w0, w1):
        for h in graph[w0]:
            if h[0] == w1:
                return h[1]

    def get_oracle_actions_onestep(sub_graph, stack, buffer, deque, actions):
        b0 = buffer[-1] if len(buffer) > 0 else -1
        s0 = stack[-1] if len(stack) > 0 else -1

        if s0 > 0 and has_head(s0, b0):
            if not has_unfound_child(s0) and not has_other_head(s0):
                actions.append("LR:" + get_arc_label(s0, b0))
                stack.pop()
                sub_graph[s0][b0] = True
                return
            else:
                actions.append("LP:" + get_arc_label(s0, b0))
                deque.append(stack.pop())
                sub_graph[s0][b0] = True
                return

        elif s0 >= 0 and has_head(b0, s0):
            if not has_other_child_in_stack(stack, b0) and not has_other_head_in_stack(stack, b0):
                actions.append("RS:" + get_arc_label(b0, s0))
                while len(deque) != 0:
                    stack.append(deque.pop())
                stack.append(buffer.pop())
                sub_graph[b0][s0] = True
                return

            elif s0 > 0:
                actions.append("RP:" + get_arc_label(b0, s0))
                deque.append(stack.pop())
                sub_graph[b0][s0] = True
                return

        elif len(buffer) != 0 and not has_other_head_in_stack(stack, b0) \
                and not has_other_child_in_stack(stack, b0):
            actions.append("NS")
            while len(deque) != 0:
                stack.append(deque.pop())
            stack.append(buffer.pop())
            return

        elif s0 > 0 and not has_unfound_child(s0) and not lack_head(s0):
            actions.append("NR")
            stack.pop()
            return

        elif s0 > 0:
            actions.append("NP")
            deque.append(stack.pop())
            return

        else:
            actions.append('-E-')
            print('"error in oracle!"')
            return

    while len(buffer) != 0:
        get_oracle_actions_onestep(sub_graph, stack, buffer, deque, actions)

    return actions
