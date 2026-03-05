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
    모든 노드 방문 여부와 상관없이, 무게 및 차량 대수 제약 내에서 
    가능한 경로 세트를 즉시 생성하여 반환 (학습용 최적화)
    """
    capacity_limit = parser.vehicle_info['capacity']
    max_vehicles = parser.vehicle_info['count']
    customer_ids = list(range(1, parser.num_customers + 1))
    
    valid_route_sets = []

    while len(valid_route_sets) < num_generated_sets:
        random.shuffle(customer_ids) # 매번 다른 아이템 시퀀스 제공
        
        current_set = []
        current_route = []
        current_load = 0
        
        for customer in customer_ids:
            demand = parser.nodes[customer]['demand']
            
            # 1. 현재 차량에 추가 가능한 경우
            if current_load + demand <= capacity_limit:
                current_route.append(customer)
                current_load += demand
            else:
                # 2. 용량 초과 시 현재 경로를 세트에 추가
                current_set.append(current_route)
                
                # 3. 차량 대수를 모두 채웠다면 중단 (모든 노드 방문 안 해도 됨)
                if len(current_set) >= max_vehicles:
                    current_route = [] # 마지막 경로는 추가하지 않음
                    break
                
                # 4. 새 차량 시작
                current_route = [customer]
                current_load = demand
        
        # 남은 경로가 있고 차량 대수 여유가 있다면 추가
        if current_route and len(current_set) < max_vehicles:
            current_set.append(current_route)
            
        # 형식 유지를 위해 부족한 차량 대수는 빈 리스트로 채움
        while len(current_set) < max_vehicles:
            current_set.append([])
            
        valid_route_sets.append(current_set)
            
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