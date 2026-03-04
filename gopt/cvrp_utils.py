import numpy as np
import random

class CVRPParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.vehicle_info = {} 
        self.nodes = {}        
        self.items = {}        
        self.num_customers = 0
        
        self._parse()

    def _parse(self):
        with open(self.filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        idx = 0
        while idx < len(lines):
            line = lines[idx]
            
            if "number of customers" in line:
                self.num_customers = int(line.split()[0])
            elif "number of vehicles" in line:
                self.vehicle_info['count'] = int(line.split()[0])
            elif "Capacity - height - width - length" in line:
                idx += 1
                vals = list(map(int, lines[idx].split()))
                self.vehicle_info['capacity'] = vals[0] 
                self.vehicle_info['height'] = vals[1]
                self.vehicle_info['width'] = vals[2]
                self.vehicle_info['length'] = vals[3]

            elif "Node - x - y - demand" in line:
                for _ in range(self.num_customers + 1): 
                    idx += 1
                    node_vals = list(map(float, lines[idx].split()))
                    node_id = int(node_vals[0])
                    self.nodes[node_id] = {
                        'x': node_vals[1], 
                        'y': node_vals[2], 
                        'demand': node_vals[3] 
                    }
                    self.items[node_id] = []

            elif "Node - number of items - h - w - l" in line:
                idx += 1
                while idx < len(lines):
                    item_line = lines[idx]
                    if "Instance" in item_line: 
                        break
                    
                    parts = list(map(str, item_line.split()))
                    node_id = int(parts[0])
                    
                    part_idx = 2
                    while part_idx < len(parts):
                        h = int(parts[part_idx])
                        w = int(parts[part_idx+1])
                        l = int(parts[part_idx+2])
                        self.items[node_id].append((l, w, h))
                        part_idx += 4
                    idx += 1
                continue 
            
            idx += 1

def generate_random_routes(parser: CVRPParser, num_generated_sets: int):
    """
    무게 제약(Capacity Constraint)을 준수하며 랜덤 경로 생성
    Return: List[List[List[int]]] -> [Set_1, Set_2, ...]
    """
    capacity_limit = parser.vehicle_info['capacity']
    max_vehicles = parser.vehicle_info['count']
    
    # 0번(Depot)을 제외한 고객 노드 리스트
    customer_ids = list(range(1, parser.num_customers + 1))
    
    valid_route_sets = []
    attempts = 0
    max_attempts = num_generated_sets * 100 # 무한루프 방지

    while len(valid_route_sets) < num_generated_sets and attempts < max_attempts:
        attempts += 1
        
        # 1. 고객 순서 랜덤 셔플
        random.shuffle(customer_ids)
        
        current_set = []     # 이번 세트의 모든 차량 경로들
        current_route = []   # 현재 채우고 있는 차량의 경로
        current_load = 0     # 현재 차량의 적재 무게
        
        possible_with_shuffle = True
        
        for customer in customer_ids:
            demand = parser.nodes[customer]['demand']
            
            # 노드 하나가 차량 용량을 넘는 경우 (데이터 오류가 아니면 불가능)
            if demand > capacity_limit:
                print(f"Error: Customer {customer} demand {demand} > Capacity {capacity_limit}")
                possible_with_shuffle = False
                break

            # 현재 차량에 더 담을 수 있는지 확인
            if current_load + demand <= capacity_limit:
                current_route.append(customer)
                current_load += demand
            else:
                # 용량 초과 -> 현재 경로 마감하고 새 차량 배차
                current_set.append(current_route)
                
                # 새 차량 시작
                current_route = [customer]
                current_load = demand
                
                # 차량 대수 제한 확인
                # 현재 마감된 차량 수 + 지금 막 시작한 차량(1) > 최대 차량 대수
                if len(current_set) + 1 > max_vehicles:
                    possible_with_shuffle = False # 이 셔플 순서로는 차량 대수 부족
                    break
        
        if possible_with_shuffle:
            # 마지막 남은 경로 추가
            if current_route:
                current_set.append(current_route)
            
            # 남는 차량이 있다면 빈 리스트로 채움 (형식 유지용)
            while len(current_set) < max_vehicles:
                current_set.append([])
                
            valid_route_sets.append(current_set)
            
    if len(valid_route_sets) < num_generated_sets:
        print(f"Warning: Only generated {len(valid_route_sets)} valid sets out of {num_generated_sets} requested.")
        
    return valid_route_sets

def get_items_for_route_reversed(parser: CVRPParser, route: list):
    """
    경로의 역순(LIFO)으로 아이템 리스트 반환
    """
    loading_sequence = []
    for node_id in reversed(route):
        node_items = parser.items.get(node_id, [])
        loading_sequence.extend(node_items)
    return loading_sequence

# --- 테스트 코드 (검증용) ---
if __name__ == "__main__":
    # 파일명을 실제 파일명으로 변경하세요
    parser = CVRPParser('3l_cvrp01.txt')
    print(f"Capacity Limit: {parser.vehicle_info['capacity']}")
    
    routes = generate_random_routes(parser, 1)
    
    if routes:
        print("\nGenerated Route Set 1:")
        for v_idx, route in enumerate(routes[0]):
            load = sum([parser.nodes[n]['demand'] for n in route])
            print(f"  Vehicle {v_idx+1}: {route} | Load: {load}/{parser.vehicle_info['capacity']} | Valid: {load <= parser.vehicle_info['capacity']}")