# from typing import Optional
# import copy

# from .container import Container
# import numpy as np
# import gymnasium as gym
# from gymnasium import spaces
# from .cutCreator import CuttingBoxCreator
# from .binCreator import RandomBoxCreator, LoadBoxCreator, BoxCreator, CVRPBoxCreator
# from render import VTKRender

# class PackingEnv(gym.Env):
#     def __init__(
#         self,
#         container_size=(10, 10, 10),
#         item_set=None, 
#         data_name=None, 
#         load_test_data=False,
#         enable_rotation=False,
#         data_type="random",
#         reward_type=None,
#         action_scheme="heightmap",
#         k_placement=100,
#         is_render=False,
#         is_hold_on=False,
#         cvrp_parsers=None,
#         **kwags
#     ) -> None:
#         self.bin_size = container_size
#         self.area = int(self.bin_size[0] * self.bin_size[1])
#         # packing state
#         self.container = Container(*self.bin_size, rotation=enable_rotation)
#         self.can_rotate = enable_rotation   
#         self.reward_type = reward_type
#         self.action_scheme = action_scheme
#         self.k_placement = k_placement
#         self.max_items = 3 
#         self.locked_heightmap = None 
#         self.failed_actions = set() # 실패한 행동 인덱스 기록

#         if action_scheme == "EMS":
#             self.candidates = np.zeros((self.k_placement, 6), dtype=np.int32)
#         else:
#             self.candidates = np.zeros((self.k_placement, 3), dtype=np.int32)

#         # Generator for train/test data
#         if not load_test_data:
#             if data_type != "cvrp":
#                 assert item_set is not None

#             if data_type == "random":
#                 print(f"using items generated randomly")
#                 self.box_creator = RandomBoxCreator(item_set)  
#             if data_type == "cut":
#                 print(f"using items generated through cutting method")
#                 low = list(item_set[0])
#                 up = list(item_set[-1])
#                 low.extend(up)
#                 self.box_creator = CuttingBoxCreator(container_size, low, self.can_rotate)
#             if data_type == "cvrp":
#                 print(f"using items generated from CVRP routes")
#                 assert cvrp_parsers is not None
#                 self.box_creator = CVRPBoxCreator(cvrp_parsers)

#             assert isinstance(self.box_creator, BoxCreator)
#         if load_test_data:
#             print(f"use box dataset: {data_name}")
#             self.box_creator = LoadBoxCreator(data_name)

#         self.test = load_test_data

#         # for rendering
#         if is_render:
#             self.renderer = VTKRender(container_size, auto_render=not is_hold_on)
#         self.render_box = None
        
#         self._set_space()

#     def _set_space(self) -> None:
#         obs_len = self.area + (self.max_items * 6) 
#         obs_len += self.k_placement * 6
        
#         self.action_space = spaces.Discrete(self.max_items * 2 * self.k_placement) 
#         self.observation_space = spaces.Dict(
#             {
#                 "obs": spaces.Box(low=0, high=max(self.bin_size), shape=(obs_len, )),
#                 "mask": spaces.Box(low=0, high=1, shape=(self.max_items * 2 * self.k_placement,), dtype=np.int8)
#             }
#         )

#     def get_box_ratio(self, box_to_place):
#         return (box_to_place[0] * box_to_place[1] * box_to_place[2]) / (
#                 self.container.dimension[0] * self.container.dimension[1] * self.container.dimension[2])

#     @property
#     def next_boxes(self) -> list:
#         return self.box_creator.preview(self.max_items)
    
#     @property
#     def cur_observation(self):
#         hmap = self.container.heightmap
#         boxes = self.next_boxes
        
#         if self.action_scheme == "EMS":
#             base_candidates, _ = self.container.candidate_from_EMS([1, 1, 1], self.k_placement)
#         else:
#             base_candidates, _ = self.container.candidate_from_heightmap([1, 1, 1], self.k_placement)

#         self.candidates = np.zeros_like(self.candidates)
#         if len(base_candidates) != 0:
#             self.candidates[0:len(base_candidates)] = base_candidates

#         all_masks = np.zeros((self.max_items, 2, self.k_placement), dtype=np.int8)
        
#         for i, box in enumerate(boxes):
#             if sum(box) == 0:
#                 continue
            
#             _, mask = self.get_possible_position(box, pre_candidates=base_candidates)
#             all_masks[i, :, :mask.shape[1]] = mask

#         flat_mask = all_masks.reshape(-1)
        
#         # 실패한 행동들은 마스킹 처리하여 다시 고르지 않게 함
#         for act in self.failed_actions:
#             if act < len(flat_mask):
#                 flat_mask[act] = 0

#         sizes = []
#         for box in boxes:
#             sizes.extend([box[0], box[1], box[2], box[1], box[0], box[2]])
            
#         obs = np.concatenate((hmap.reshape(-1), np.array(sizes).reshape(-1), self.candidates.reshape(-1)))
        
#         return {
#             "obs": obs, 
#             "mask": flat_mask
#         }

#     def get_possible_position(self, next_box, pre_candidates=None):
#         if pre_candidates is not None:
#             candidates = pre_candidates
#             mask = np.zeros((2, len(candidates)), dtype=np.int8)
#             # EMS 체크
#             for i, ems in enumerate(candidates):
#                 if self.container.check_box_ems(next_box, ems) > -1:
#                     mask[0, i] = 1
#                 if self.can_rotate:
#                     rotated_box = [next_box[1], next_box[0], next_box[2]]
#                     if self.container.check_box_ems(rotated_box, ems) > -1:
#                         mask[1, i] = 1
#         else:
#             if self.action_scheme == "EMS":
#                 candidates, mask = self.container.candidate_from_EMS(next_box, self.k_placement)
#             else:
#                 candidates = self.container.candidate_from_heightmap(next_box, self.k_placement)
#                 mask = np.ones((2, len(candidates)), dtype=np.int8)

#         if hasattr(self, 'locked_heightmap') and self.locked_heightmap is not None:
#             hmap_for_lifo = self.locked_heightmap
#         else:
#             hmap_for_lifo = self.container.heightmap
            
#         for i, ems in enumerate(candidates):
#             x_start, y_start, z_base = int(ems[0]), int(ems[1]), int(ems[2])

#             for rot in range(2):
#                 if mask[rot, i] == 0: continue 

#                 curr_size_x = next_box[0] if rot == 0 else next_box[1]
#                 curr_size_y = next_box[1] if rot == 0 else next_box[0]
                
#                 if x_start + curr_size_x >= hmap_for_lifo.shape[0]:
#                     continue

#                 y_end = min(y_start + curr_size_y, hmap_for_lifo.shape[1])
                
#                 path_area = hmap_for_lifo[x_start + curr_size_x:, y_start:y_end]
                
#                 if np.any(path_area > z_base):
#                     mask[rot, i] = 0

#         return candidates, mask

#     def idx2pos(self, idx):
#         item_idx = idx // (2 * self.k_placement)
#         rem = idx % (2 * self.k_placement)
        
#         rot = 1 if rem >= self.k_placement else 0
#         ems_idx = rem % self.k_placement

#         pos = self.candidates[ems_idx][:3]
#         box = self.next_boxes[item_idx]

#         if rot == 1:
#             dim = [box[1], box[0], box[2]]
#         else:
#             dim = list(box)
#         self.render_box = [dim, pos]

#         return pos, rot, dim, item_idx
    
#     def step(self, action):
#         pos, rot, size, item_idx = self.idx2pos(action)
#         box_to_place = self.next_boxes[item_idx]
 
#         # 공통 info 생성 함수
#         def create_info_and_print(is_done):
#             packed_items = len(self.container.boxes)
#             total_items = getattr(self.box_creator, 'total_route_items', 0)
#             route = getattr(self.box_creator, 'current_route', [])
#             is_success = (packed_items == total_items) and (total_items > 0)
            
#             info_dict = {
#                 'counter': packed_items, 
#                 'ratio': self.container.get_volume_ratio(),
#                 'route': str(route),
#                 'total_items': total_items,
#                 'is_success': is_success
#             }
            
#             if is_done:
#                 status = "SUCCESS" if is_success else "FAIL"
#                 print(f"✅ [Test Result] Route: {route} | Packed: {packed_items}/{total_items} | Result: {status} | Space Ratio: {info_dict['ratio']:.4f}")
#             return info_dict

#         if sum(box_to_place) == 0:
#             done = True
#             info = create_info_and_print(done)
#             return self.cur_observation, 0.0, done, False, info

#         succeeded = self.container.place_box(box_to_place, pos, rot)
        
#         if not succeeded:
#             self.failed_actions.add(action)
#             reward = -0.1 
#             self.render_box = [[0, 0, 0], [0, 0, 0]]
            
#             obs_dict = self.cur_observation
#             done = (obs_dict["mask"].sum() == 0)
#             info = create_info_and_print(done)
#             return obs_dict, reward, done, False, info

#         self.failed_actions.clear()
#         box_ratio = self.get_box_ratio(box_to_place)
#         old_node_idx = getattr(self.box_creator, 'current_node_idx', 0)
        
#         self.box_creator.drop_box(item_idx)
            
#         new_node_idx = getattr(self.box_creator, 'current_node_idx', 0)
        
#         if old_node_idx != new_node_idx:
#             self.locked_heightmap = copy.deepcopy(self.container.heightmap)

#         self.box_creator.generate_box_size()  

#         if self.reward_type == "terminal":
#             reward = 0.01
#         else:
#             reward = box_ratio
            
#         obs_dict = self.cur_observation
#         done = (obs_dict["mask"].sum() == 0)
#         info = create_info_and_print(done)

#         return obs_dict, reward, done, False, info

#     def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
#         super().reset(seed=seed)
#         self.box_creator.reset()
#         self.container = Container(*self.bin_size)
#         self.locked_heightmap = copy.deepcopy(self.container.heightmap) 
#         self.failed_actions.clear()
#         self.box_creator.generate_box_size()
#         self.candidates = np.zeros_like(self.candidates)
#         return self.cur_observation, {}
    
#     def seed(self, s=None):
#         np.random.seed(s)

#     def render(self):
#         self.renderer.add_item(self.render_box[0], self.render_box[1])

from typing import Optional
import copy

from .container import Container
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .cutCreator import CuttingBoxCreator
from .binCreator import RandomBoxCreator, LoadBoxCreator, BoxCreator, CVRPBoxCreator
from render import VTKRender

class PackingEnv(gym.Env):
    def __init__(
        self,
        container_size=(10, 10, 10),
        item_set=None, 
        data_name=None, 
        load_test_data=False,
        enable_rotation=False,
        data_type="random",
        reward_type=None,
        action_scheme="heightmap",
        k_placement=100,
        is_render=False,
        is_hold_on=False,
        cvrp_parsers=None,
        **kwags
    ) -> None:
        self.bin_size = container_size
        self.area = int(self.bin_size[0] * self.bin_size[1])
        # packing state
        self.container = Container(*self.bin_size, rotation=enable_rotation)
        self.can_rotate = enable_rotation   
        self.reward_type = reward_type
        self.action_scheme = action_scheme
        self.k_placement = k_placement
        self.max_items = 3 
        self.locked_heightmap = None 
        self.failed_actions = set() # 실패한 행동 인덱스 기록

        if action_scheme == "EMS":
            self.candidates = np.zeros((self.k_placement, 6), dtype=np.int32)
        else:
            self.candidates = np.zeros((self.k_placement, 3), dtype=np.int32)

        # Generator for train/test data
        if not load_test_data:
            if data_type != "cvrp":
                assert item_set is not None

            if data_type == "random":
                print(f"using items generated randomly")
                self.box_creator = RandomBoxCreator(item_set)  
            if data_type == "cut":
                print(f"using items generated through cutting method")
                low = list(item_set[0])
                up = list(item_set[-1])
                low.extend(up)
                self.box_creator = CuttingBoxCreator(container_size, low, self.can_rotate)
            if data_type == "cvrp":
                print(f"using items generated from CVRP routes")
                assert cvrp_parsers is not None
                self.box_creator = CVRPBoxCreator(cvrp_parsers)

            assert isinstance(self.box_creator, BoxCreator)
        if load_test_data:
            print(f"use box dataset: {data_name}")
            self.box_creator = LoadBoxCreator(data_name)

        self.test = load_test_data

        # for rendering
        if is_render:
            self.renderer = VTKRender(container_size, auto_render=not is_hold_on)
        self.render_box = None
        
        self._set_space()

    def _set_space(self) -> None:
        obs_len = self.area + (self.max_items * 6) 
        obs_len += self.k_placement * 6
        
        self.action_space = spaces.Discrete(self.max_items * 2 * self.k_placement) 
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(low=0, high=max(self.bin_size), shape=(obs_len, )),
                "mask": spaces.Box(low=0, high=1, shape=(self.max_items * 2 * self.k_placement,), dtype=np.int8)
            }
        )

    def get_box_ratio(self, box_to_place):
        return (box_to_place[0] * box_to_place[1] * box_to_place[2]) / (
                self.container.dimension[0] * self.container.dimension[1] * self.container.dimension[2])

    @property
    def next_boxes(self) -> list:
        return self.box_creator.preview(self.max_items)
    
    @property
    def cur_observation(self):
        hmap = self.container.heightmap
        boxes = self.next_boxes
        
        if self.action_scheme == "EMS":
            base_candidates, _ = self.container.candidate_from_EMS([1, 1, 1], self.k_placement)
        else:
            base_candidates, _ = self.container.candidate_from_heightmap([1, 1, 1], self.k_placement)

        self.candidates = np.zeros_like(self.candidates)
        if len(base_candidates) != 0:
            self.candidates[0:len(base_candidates)] = base_candidates

        all_masks = np.zeros((self.max_items, 2, self.k_placement), dtype=np.int8)
        
        for i, box in enumerate(boxes):
            if sum(box) == 0:
                continue
            
            _, mask = self.get_possible_position(box, pre_candidates=base_candidates)
            all_masks[i, :, :mask.shape[1]] = mask

        flat_mask = all_masks.reshape(-1)
        
        # 실패한 행동들은 마스킹 처리하여 다시 고르지 않게 함
        for act in self.failed_actions:
            if act < len(flat_mask):
                flat_mask[act] = 0

        sizes = []
        for box in boxes:
            sizes.extend([box[0], box[1], box[2], box[1], box[0], box[2]])
            
        obs = np.concatenate((hmap.reshape(-1), np.array(sizes).reshape(-1), self.candidates.reshape(-1)))
        
        return {
            "obs": obs, 
            "mask": flat_mask
        }

    def get_possible_position(self, next_box, pre_candidates=None):
        if pre_candidates is not None:
            candidates = pre_candidates
            mask = np.zeros((2, len(candidates)), dtype=np.int8)
            # EMS 체크
            for i, ems in enumerate(candidates):
                if self.container.check_box_ems(next_box, ems) > -1:
                    mask[0, i] = 1
                if self.can_rotate:
                    rotated_box = [next_box[1], next_box[0], next_box[2]]
                    if self.container.check_box_ems(rotated_box, ems) > -1:
                        mask[1, i] = 1
        else:
            if self.action_scheme == "EMS":
                candidates, mask = self.container.candidate_from_EMS(next_box, self.k_placement)
            else:
                candidates = self.container.candidate_from_heightmap(next_box, self.k_placement)
                mask = np.ones((2, len(candidates)), dtype=np.int8)

        if hasattr(self, 'locked_heightmap') and self.locked_heightmap is not None:
            hmap_for_lifo = self.locked_heightmap
        else:
            hmap_for_lifo = self.container.heightmap
            
        for i, ems in enumerate(candidates):
            x_start, y_start, z_base = int(ems[0]), int(ems[1]), int(ems[2])

            for rot in range(2):
                if mask[rot, i] == 0: continue 

                curr_size_x = next_box[0] if rot == 0 else next_box[1]
                curr_size_y = next_box[1] if rot == 0 else next_box[0]
                
                if x_start + curr_size_x >= hmap_for_lifo.shape[0]:
                    continue

                y_end = min(y_start + curr_size_y, hmap_for_lifo.shape[1])
                
                path_area = hmap_for_lifo[x_start + curr_size_x:, y_start:y_end]
                
                if np.any(path_area > z_base):
                    mask[rot, i] = 0

        return candidates, mask

    def idx2pos(self, idx):
        item_idx = idx // (2 * self.k_placement)
        rem = idx % (2 * self.k_placement)
        
        rot = 1 if rem >= self.k_placement else 0
        ems_idx = rem % self.k_placement

        pos = self.candidates[ems_idx][:3]
        box = self.next_boxes[item_idx]

        if rot == 1:
            dim = [box[1], box[0], box[2]]
        else:
            dim = list(box)
        self.render_box = [dim, pos]

        return pos, rot, dim, item_idx
    
    def step(self, action):
        pos, rot, size, item_idx = self.idx2pos(action)
        box_to_place = self.next_boxes[item_idx]
 
        # 공통 info 생성 함수
        def create_info_and_print(is_done):
            packed_items = len(self.container.boxes)
            total_items = getattr(self.box_creator, 'total_route_items', 0)
            route = getattr(self.box_creator, 'current_route', [])
            is_success = (packed_items == total_items) and (total_items > 0)
            
            info_dict = {
                'counter': packed_items, 
                'ratio': self.container.get_volume_ratio(),
                'route': str(route),
                'total_items': total_items,
                'is_success': is_success
            }
            
            if is_done:
                status = "SUCCESS" if is_success else "FAIL"
                #print(f"✅ [Test Result] Route: {route} | Packed: {packed_items}/{total_items} | Result: {status} | Space Ratio: {info_dict['ratio']:.4f}")
            return info_dict

        if sum(box_to_place) == 0:
            done = True
            info = create_info_and_print(done)
            return self.cur_observation, 0.0, done, False, info

        succeeded = self.container.place_box(box_to_place, pos, rot)
        
        if not succeeded:
            self.failed_actions.add(action)
            reward = -0.1 
            self.render_box = [[0, 0, 0], [0, 0, 0]]
            
            obs_dict = self.cur_observation
            done = (obs_dict["mask"].sum() == 0)
            info = create_info_and_print(done)
            return obs_dict, reward, done, False, info

        self.failed_actions.clear()
        box_ratio = self.get_box_ratio(box_to_place)
        old_node_idx = getattr(self.box_creator, 'current_node_idx', 0)
        
        self.box_creator.drop_box(item_idx)
            
        new_node_idx = getattr(self.box_creator, 'current_node_idx', 0)
        
        if old_node_idx != new_node_idx:
            self.locked_heightmap = copy.deepcopy(self.container.heightmap)

        self.box_creator.generate_box_size()  

        if self.reward_type == "terminal":
            reward = 0.01
        else:
            reward = box_ratio
            
        obs_dict = self.cur_observation
        done = (obs_dict["mask"].sum() == 0)
        info = create_info_and_print(done)

        return obs_dict, reward, done, False, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.box_creator.reset()
        self.container = Container(*self.bin_size)
        self.locked_heightmap = copy.deepcopy(self.container.heightmap) 
        self.failed_actions.clear()
        self.box_creator.generate_box_size()
        self.candidates = np.zeros_like(self.candidates)
        return self.cur_observation, {}
    
    def seed(self, s=None):
        np.random.seed(s)

    def render(self):
        self.renderer.add_item(self.render_box[0], self.render_box[1])

class PackingEnvNoSup(PackingEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.container = Container(*self.bin_size, rotation=self.can_rotate, support_constraint=False)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options) # <- 수정: 부모 클래스의 reset을 안전하게 호출
        
        self.box_creator.reset()
        # 부모가 만든 container를 덮어씌움
        self.container = Container(*self.bin_size, rotation=self.can_rotate, support_constraint=False)
        self.locked_heightmap = copy.deepcopy(self.container.heightmap) 
        self.failed_actions.clear()
        self.box_creator.generate_box_size()
        self.candidates = np.zeros_like(self.candidates)
        
        return self.cur_observation, {}