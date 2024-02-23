import inspect
import yaml
from anytree import Node, RenderTree
from torch.nn import Sequential
import networkx as nx
import matplotlib.pyplot as plt


class TorchModelAnalyzer:
    def __init__(self, model):
        self.model_tree = self.__model_to_tree(model)

    def __model_to_tree(self, layer, parent=None):
        layer_name = layer.__class__.__name__
        if (isinstance(layer, Sequential)) and layer_name != "Sequential":
            layer_name += "(Sequential)"

        layer_name_org = layer.__class__.__name__
        layer_params = (
            self.__get_layer_params(layer) if len(layer._modules) == 0 else ""
        )  # child가 없는 leaf인 경우에만 param 체크

        forward_function_src = inspect.getsource(layer.forward)
        init_function_src = inspect.getsource(layer.__init__)

        # 디버깅을 위해 일단 layer 인스턴스도 노드에 저장함
        cur_node = Node(
            layer_name,
            parent,
            layer_instance=layer,
            layer_name_org=layer_name_org,
            layer_params=layer_params,
            forward_function_src=forward_function_src,
            init_function_src=init_function_src,
        )

        for key, module in layer._modules.items():
            self.__model_to_tree(module, cur_node)
        return cur_node

    def print_model_tree(self, depth_limit=None, show_detail=False):
        for pre, fill, node in RenderTree(self.model_tree):
            if not (depth_limit is None):
                if node.depth > depth_limit:
                    continue
            layer_info = ""
            if show_detail:
                layer_info = (
                    f" - {node.layer_params}" if len(node.children) == 0 else ""
                )
            print(f"{pre}{node.name}{layer_info}")

    def save_model_modules_info(self, savepath):
        # 모델 구조 입력
        model_tree_list = []

        for pre, fill, node in RenderTree(self.model_tree):
            layer_info = f" - {node.layer_params}" if len(node.children) == 0 else ""
            model_tree_list.append(f"{pre}{node.name}{layer_info}")

        # 모델 코드 입력
        dic = {}

        # node 탐색
        for pre, fill, node in RenderTree(self.model_tree):
            # 노드가 없는 경우에만 데이터 추가
            if not (node.layer_name_org in dic.keys()):
                nodeinfo = {}
                nodeinfo["name"] = node.name
                nodeinfo["params"] = node.layer_params
                nodeinfo["init_code"] = node.init_function_src
                nodeinfo["forward_code"] = node.forward_function_src

                dic[node.layer_name_org] = nodeinfo.copy()

        # 노드 출력
        with open(savepath, "w") as f:
            # 모델 구조 출력
            f.write("Model Tree \n\n")
            for model_tree_line in model_tree_list:
                f.write(str(model_tree_line))
                f.write("\n")

            # 모델 코드 출력
            f.write("\n\nModel Codes \n\n")
            for i, (key, value) in enumerate(dic.items()):
                f.write(f"======================================")
                f.write("\n")

                f.write(f"module {i}")
                f.write("\n")

                f.write(f"Name : {value['name']}")
                f.write("\n")

                f.write(f"__init__() : ")
                f.write("\n")
                f.write(f"{str(value['init_code'])}")
                f.write("\n")

                f.write(f"forward() : ")
                f.write("\n")
                f.write(f"{str(value['forward_code'])}")
                f.write("\n")

                f.write(f"======================================")
                f.write("\n")

        return dic

    def __get_paramname_from_index(self, func, index):
        params = inspect.signature(func).parameters
        name = ""
        for i, param in enumerate(params):
            if i == index:
                name = str(param)
        return name

    def __str_to_orgtype(self, string):
        return_value = string

        try:
            return_value = int(string)
            return return_value
        except:
            pass

        try:
            return_value = float(string)
            return return_value
        except:
            pass

        try:
            if (string == "True") or (string == "False"):
                return_value = True if string == "True" else False
                return return_value
        except:
            pass

        return return_value

    def __get_layer_params(self, layer):
        """Layer의 생성자에 필요한 파라미터 추출

        Args:
            layer (_type_): nn.module

        Returns:
            파라미터 dictionary
        """
        func = layer.__init__
        raw_string = str(layer)[str(layer).find("(") + 1 : -1]

        str_list = raw_string.split(", ") if raw_string != "" else []

        raw_param_strings = []
        for i, chunk in enumerate(str_list):
            if "(" in chunk:
                raw_param_strings.append(f"{str_list[i]}, {str_list[i+1]}")
            elif ")" in chunk:
                continue
            else:
                raw_param_strings.append(chunk)

        func_params = {}
        for i, param in enumerate(raw_param_strings):
            key, value = "", ""
            if "=" in param:
                key = param.split("=")[0]
                value = param.split("=")[1]
            else:  # named parameter가 아닌경우
                key = self.__get_paramname_from_index(
                    func, i
                )  # 함수 코드를 찾아서 직접 변수 이름을 찾아옴
                value = param

            if "," in value:  # value가 tuple인경우
                value = eval(value)[0]  # 첫번째 값만 받아옴

            value = self.__str_to_orgtype(value)
            func_params[key] = value

        return func_params

    def get_only_leaf_nodes(self) -> list:
        result_list = []
        for _, _, node in RenderTree(self.model_tree):
            if len(node.children) == 0:
                result_list.append(node)

        return result_list

    def model_leafs_to_leopard_yaml(self, path=None):
        result_dict = {"Layers": {}}

        leafs = self.get_only_leaf_nodes()

        prev_node = ["None"]
        for index, node in enumerate(leafs):
            # 변수 설정
            layer_name_org = node.layer_name_org
            layer_name_with_index = f"{node.layer_name_org}_{str(index).zfill(3)}"
            params = node.layer_params

            # dict 내용 입력
            content_of_dict = {}
            content_of_dict["type"] = layer_name_org

            if layer_name_org == "Add":
                temp_index = index - 16
                additional_layer_name = (
                    f"{leafs[temp_index].layer_name_org}_{str(temp_index).zfill(3)}"
                )
                prev_node.append(additional_layer_name)
            elif layer_name_org == "EwMul":
                prev_node = []
                temp_index = index - 6
                additional_layer_name = (
                    f"{leafs[temp_index].layer_name_org}_{str(temp_index).zfill(3)}"
                )
                prev_node.append(additional_layer_name)

                temp_index = index - 1
                additional_layer_name = (
                    f"{leafs[temp_index].layer_name_org}_{str(temp_index).zfill(3)}"
                )
                prev_node.append(additional_layer_name)

            content_of_dict["input_link"] = prev_node
            if len(params) != 0:
                content_of_dict["params"] = params

            # dict에 넣기
            result_dict["Layers"][layer_name_with_index] = content_of_dict.copy()

            # prev 설정
            prev_node = [layer_name_with_index]

        if not (path is None):
            with open(path, "w+") as f:
                yaml.dump(
                    result_dict,
                    f,
                    allow_unicode=True,
                    default_flow_style=False,
                    sort_keys=False,
                )

        return result_dict

    def draw_graph_from_yaml(self, yaml_path, png_path):
        with open(yaml_path, "r") as f:
            table = yaml.load(f, Loader=yaml.FullLoader)

        G = nx.Graph()
        G.add_nodes_from(list(table["Layers"].keys()))  # 노드 설정
        for i, layer in enumerate(table["Layers"]):
            input_links = table["Layers"][layer]["input_link"]
            for input_link in input_links:
                G.add_edge(layer, input_link)  # 노드와 엣지 설정

        pos = nx.kamada_kawai_layout(G)  # 그래프의 레이아웃 (노드, 엣지 위치 설정)
        # pos = nx.circular_layout(G)
        # pos = nx.tree_graph(G)
        nx.draw(
            G,
            pos=pos,
            ax=None,
            with_labels=True,
            font_size=2,
            node_size=4,
            width=0.5,
            node_color="lightgreen",
        )  # 그리기
        plt.savefig(png_path, dpi=500)
