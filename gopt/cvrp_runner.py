import sys
import os
import numpy as np

# [수정 1] container와 binCreator의 경로를 envs.Packing 패키지에서 가져오도록 변경
# container.py와 binCreator.py가 envs/Packing 폴더 안에 있기 때문입니다.
try:
    from envs.Packing.container import Container
    from envs.Packing.binCreator import BoxCreator
except ImportError:
    # 만약 패키지 인식이 안 될 경우를 대비해 경로를 강제로 추가
    sys.path.append(os.path.join(os.path.dirname(__file__), 'envs', 'Packing'))
    from container import Container
    from binCreator import BoxCreator

# cvrp_utils는 같은 폴더(최상위)에 있으므로 그대로 import
from cvrp_utils import CVRPParser, generate_random_routes, get_items_for_route_reversed

# [수정 2] RouteBoxCreator 클래스를 여기서 직접 정의 (binCreator.py 수정 불필요)
class RouteBoxCreator(BoxCreator):
    """
    CVRP 경로에 의해 결정된 아이템 리스트를 순차적으로 제공하는 클래스
    """
    def __init__(self, item_sequence):
        super().__init__()
        self.all_items = item_sequence # List of (l, w, h)
        self.reset()

    def reset(self):
        super().reset()
        self.box_index = 0
        self.box_list.clear()
        
    def generate_box_size(self, **kwargs):
        """
        리스트에 있는 다음 아이템을 가져와 box_list에 추가
        """
        if self.box_index < len(self.all_items):
            self.box_list.append(self.all_items[self.box_index])
            self.box_index += 1
        else:
            # 더 이상 아이템이 없으면 (0,0,0)으로 종료 신호
            self.box_list.append((0, 0, 0))

def check_route_feasibility(container_dims, item_sequence, action_scheme="EMS"):
    """
    단일 차량 경로에 대한 패킹 가능 여부 확인
    학습된 모델이 없으므로 Greedy하게 EMS/EP 등을 사용하여 적재 시도
    """
    L, W, H = container_dims
    # GOPT Container 초기화
    container = Container(length=L, width=W, height=H, rotation=True)
    
    # Box Creator 초기화
    creator = RouteBoxCreator(item_sequence)
    
    # 모든 아이템에 대해 패킹 시도
    # item_sequence 전체 길이만큼 반복
    total_items = len(item_sequence)
    
    for _ in range(total_items):
        creator.generate_box_size() # 박스 생성(큐에 추가)
        if not creator.box_list: break
        
        # 현재 배치해야 할 박스 (next_box)
        next_box = creator.preview(1)[0]
        if next_box == (0,0,0): break
        
        # 1. 후보 위치 생성 (Action Masking)
        if action_scheme == "EMS":
            candidates, mask = container.candidate_from_EMS(next_box, max_n=100)
        elif action_scheme == "EP":
            candidates, mask = container.candidate_from_EP(next_box, max_n=100)
        else:
            candidates = container.candidate_from_heightmap(next_box, max_n=100)
            
        # 2. 배치 결정 (Policy 부분 - Greedy Heuristic)
        succeeded = False
        if len(candidates) > 0:
            best_action = None
            rot_flag = 0
            
            # 단순 Greedy: 리스트 앞쪽(Bottom-Left-Deepest)이면서 가능한 것 선택
            for i in range(len(candidates)):
                # 회전 안 함
                if mask[0][i] == 1: 
                    best_action = candidates[i]
                    rot_flag = 0
                    break
                # 회전 함
                if mask[1][i] == 1:
                    best_action = candidates[i]
                    rot_flag = 1
                    break
            
            if best_action is not None:
                pos = [best_action[0], best_action[1], best_action[2]]
                # 배치 실행
                succeeded = container.place_box(next_box, pos, rot_flag)
        
        if succeeded:
            creator.drop_box() # 배치 성공 시 목록에서 제거
        else:
            # 배치 실패 -> 이 경로는 불가능
            return False, f"Failed to pack item {next_box}"

    return True, "Success"

def main():
    # 1. 설정
    # 파일 경로를 사용자의 환경에 맞게 수정 (3L_CVRP 폴더가 있다면 포함)
    data_file = '3l_cvrp01.txt' 
    if not os.path.exists(data_file):
        # 만약 3L_CVRP 폴더 안에 있다면 경로 변경 시도
        if os.path.exists(os.path.join('3L_CVRP', data_file)):
            data_file = os.path.join('3L_CVRP', data_file)
    
    num_random_sets = 5 # 생성할 랜덤 경로 세트 수
    
    # 2. 데이터 파싱
    print(f"Loading data from: {data_file}")
    parser = CVRPParser(data_file)
    print(f"Loaded: {parser.num_customers} customers, {parser.vehicle_info['count']} vehicles")
    
    container_dims = (
        parser.vehicle_info['length'], 
        parser.vehicle_info['width'], 
        parser.vehicle_info['height']
    )
    print(f"Vehicle Dimensions (L, W, H): {container_dims}")

    # 3. 경로 생성 (Route Sets)
    route_sets = generate_random_routes(parser, num_random_sets)
    
    # 4. 검증 루프
    for set_idx, route_set in enumerate(route_sets):
        print(f"\n--- Checking Route Set {set_idx + 1} ---")
        set_valid = True
        
        for v_idx, route in enumerate(route_set):
            if not route: continue # 빈 경로는 패스
            
            # 경로의 역순으로 아이템 로드 (LIFO)
            items = get_items_for_route_reversed(parser, route)
            
            # 적재 시뮬레이션
            is_possible, msg = check_route_feasibility(container_dims, items, action_scheme="EMS")
            
            print(f"  Vehicle {v_idx+1} (Nodes: {route}): {msg}")
            
            if not is_possible:
                print(f"  -> Route Set {set_idx + 1} INVALID. Skipping to next set.")
                set_valid = False
                break 
        
        if set_valid:
            print(f"==> Route Set {set_idx + 1} is FEASIBLE! <==")
            break
        else:
            print(f"==> Route Set {set_idx + 1} Failed.")

if __name__ == "__main__":
    main()