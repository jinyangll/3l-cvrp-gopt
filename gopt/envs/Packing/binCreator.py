import numpy as np
import copy
import torch
import random
# cvrp_utils.py가 같은 폴더에 있어야 합니다.
from cvrp_utils import CVRPParser, generate_random_routes, get_items_for_route_reversed


class BoxCreator(object):
    def __init__(self):
        self.box_list = []  # generated box list

    def reset(self):
        self.box_list.clear()

    def generate_box_size(self, **kwargs):
        pass

    def preview(self, length):
        """
        :param length:
        :return: list
        """
        while len(self.box_list) < length:
            self.generate_box_size()
        return copy.deepcopy(self.box_list[:length])

    def drop_box(self, item_idx=0):
        # [수정] 여러 아이템 중 특정 인덱스를 선택해 버릴 수 있도록 파라미터 추가 (기본값 0)
        assert len(self.box_list) > item_idx
        self.box_list.pop(item_idx)


class RandomBoxCreator(BoxCreator):
    default_box_set = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                default_box_set.append((2 + i, 2 + j, 2 + k))

    def __init__(self, box_size_set=None):
        super().__init__()
        self.box_set = box_size_set
        if self.box_set is None:
            self.box_set = RandomBoxCreator.default_box_set
        # print(self.box_set)

    def generate_box_size(self, **kwargs):
        idx = np.random.randint(0, len(self.box_set))
        self.box_list.append(self.box_set[idx])


# load data
class LoadBoxCreator(BoxCreator):
    def __init__(self, data_name=None):  # data url
        super().__init__()  
        self.data_name = data_name
        self.index = 0
        self.box_index = 0
        self.traj_nums = len(torch.load(self.data_name))  
        print("load data set successfully, data name: ", self.data_name)

    def reset(self, index=None):
        self.box_list.clear()
        box_trajs = torch.load(self.data_name)
        self.recorder = []
        if index is None:
            self.index += 1
        else:
            self.index = index
        self.boxes = box_trajs[self.index]
        self.box_index = 0
        self.box_set = self.boxes
        self.box_set.append([10, 10, 10])

    def generate_box_size(self, **kwargs):
        if self.box_index < len(self.box_set):
            self.box_list.append(self.box_set[self.box_index])
            self.recorder.append(self.box_set[self.box_index])
            self.box_index += 1
        else:
            self.box_list.append((10, 10, 10))
            self.recorder.append((10, 10, 10))
            self.box_index += 1

class CVRPBoxCreator(BoxCreator):
    def __init__(self, cvrp_parsers: list):
        super().__init__()
        self.parsers = cvrp_parsers # 리스트로 저장
        self.parser = None # 현재 에피소드에서 사용할 파서
        self.node_items = [] 
        self.current_node_idx = 0
        
    def reset(self):
        self.node_items = []
        self.current_node_idx = 0
        
        # 유효한 경로가 생성될 때까지 파서를 다시 무작위로 뽑으며 반복
        route_sets = []
        while not route_sets:
            self.parser = random.choice(self.parsers)
            route_sets = generate_random_routes(self.parser, 1) 
            
        vehicle_routes = route_sets[0]
        
        valid_routes = [r for r in vehicle_routes if len(r) > 0]
        if not valid_routes:
            target_route = []
        else:
            target_route = random.choice(valid_routes)

        self.current_route = target_route
        self.total_route_items = 0 # 해당 경로의 총 아이템 개수 누적
            
        for node_id in reversed(target_route):
            items_in_node = self.parser.items.get(node_id, [])
            if items_in_node:
                self.node_items.append(list(items_in_node)) 
                self.total_route_items += len(items_in_node)
        
        if not self.node_items:
             self.node_items = [[(0, 0, 0)]]
             self.total_route_items = 0
             
    def preview(self, length=3):
        """ [수정] 한 스텝에서 볼 수 있는 후보 아이템 반환. 현재 노드의 아이템만 반환하며, 
            지정된 length(기본 3)보다 모자란 칸은 (0,0,0)으로 패딩합니다.
        """
        # 경로 상의 모든 노드를 다 방문했으면 더미 반환
        if self.current_node_idx >= len(self.node_items):
            return [(0, 0, 0) for _ in range(length)]
        
        cur_items = self.node_items[self.current_node_idx]
        result = []
        
        for i in range(length):
            if i < len(cur_items):
                result.append(cur_items[i])
            else:
                result.append((0, 0, 0)) # Padding
                
        return result

    def drop_box(self, item_idx=0):
        """ [수정] 에이전트가 선택한 특정 인덱스(0, 1, 2 중 하나)의 아이템을 현재 노드 리스트에서 제거.
            현재 노드의 아이템이 다 소진되면 다음 노드로 이동합니다.
        """
        if self.current_node_idx < len(self.node_items):
            cur_items = self.node_items[self.current_node_idx]
            
            # env.py에서 패딩된 인덱스는 이미 마스킹되어 오지 않겠지만 방어 코드 작성
            if item_idx < len(cur_items):
                cur_items.pop(item_idx)
            
            # 현재 노드 아이템을 모두 적재했다면 다음 노드로 인덱스 이동
            if len(cur_items) == 0:
                self.current_node_idx += 1

    def generate_box_size(self, **kwargs):
        """ [수정] CVRPBoxCreator에서는 preview가 동적으로 node_items를 참조하므로
            별도로 box_list에 append하는 기존 로직이 불필요합니다.
        """
        pass

class EvalBoxCreator(CVRPBoxCreator):
    def __init__(self, cvrp_parser: CVRPParser):
        super().__init__([cvrp_parser]) 
        self.parser = cvrp_parser 
        self.eval_route = []

    def set_route(self, route):
        self.eval_route = route

    def reset(self):
        self.node_items = []
        self.current_node_idx = 0
        self.current_route = self.eval_route
        self.total_route_items = 0
        
        for node_id in reversed(self.eval_route):
            items_in_node = self.parser.items.get(node_id, [])
            if items_in_node:
                self.node_items.append(list(items_in_node))
                self.total_route_items += len(items_in_node)
        
        if not self.node_items:
             self.node_items = [[(0, 0, 0)]]
             self.total_route_items = 0