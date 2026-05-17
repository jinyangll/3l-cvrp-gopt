import numpy as np
import random
import math

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
    아이템 개수 기반 경로 생성 (모든 노드 방문 보장)
    """
    max_vehicles = parser.vehicle_info['count']
    customer_ids = list(range(1, parser.num_customers + 1))
    
    # 1. 총 아이템 개수 계산 및 차량별 한도 설정 (올림)
    total_items = sum(len(parser.items.get(node_id, [])) for node_id in customer_ids)
    item_limit = math.ceil(total_items / max_vehicles)
    
    valid_route_sets = []

    while len(valid_route_sets) < num_generated_sets:
        random.shuffle(customer_ids)
        
        current_set = []
        current_route = []
        current_item_count = 0
        is_success = True
        
        for customer in customer_ids:
            node_items = parser.items.get(customer, [])
            node_item_count = len(node_items)
            
            # 2. 한도 초과 시 다음 차량으로 넘김 (노드 분할 금지)
            if current_item_count + node_item_count <= item_limit:
                current_route.append(customer)
                current_item_count += node_item_count
            else:
                current_set.append(current_route)
                
                # 3. 차량 대수를 초과했는데 노드가 남은 경우 실패 처리
                if len(current_set) >= max_vehicles:
                    is_success = False
                    break
                
                current_route = [customer]
                current_item_count = node_item_count
        
        if is_success:
            if current_route:
                current_set.append(current_route)
            
            # 차량 대수 유지를 위해 빈 리스트 추가
            while len(current_set) < max_vehicles:
                current_set.append([])
                
            valid_route_sets.append(current_set)
            
    return valid_route_sets

def get_items_for_route_reversed(parser: CVRPParser, route: list):
    loading_sequence = []
    for node_id in reversed(route):
        node_items = parser.items.get(node_id, [])
        loading_sequence.extend(node_items)
    return loading_sequence

def calculate_distance(n1, n2):
    return math.sqrt((n1['x'] - n2['x'])**2 + (n1['y'] - n2['y'])**2)

def generate_saving_routes(parser: CVRPParser, top_k: int = 5):
    """
    무작위성이 추가된 Savings 알고리즘.
    top_k: 병합 가능한 후보 중 상위 몇 개 내에서 랜덤하게 선택할지 결정 (기본값 5)
    """
    depot = parser.nodes[0]
    customers = {node_id: info for node_id, info in parser.nodes.items() if node_id != 0}
    max_capacity = parser.vehicle_info['capacity']  # 55
    
    # 1. 초기화: 모든 고객을 개별 경로로 설정
    routes = [[i] for i in customers.keys()]
    
    # 2. 모든 가능한 Savings 계산
    all_savings = []
    customer_ids = list(customers.keys())
    for i in range(len(customer_ids)):
        for j in range(i + 1, len(customer_ids)):
            id_i, id_j = customer_ids[i], customer_ids[j]
            d0i = calculate_distance(depot, customers[id_i])
            d0j = calculate_distance(depot, customers[id_j])
            dij = calculate_distance(customers[id_i], customers[id_j])
            s = d0i + d0j - dij
            if s > 0:
                all_savings.append((s, id_i, id_j)) 
    
    # 3. 반복적 병합 (무작위 선택 로직 포함)
    while True:
        # 현재 상태에서 병합 가능한 후보들 찾기
        available_merges = []
        for s, i, j in all_savings:
            route_i = next((r for r in routes if i in r), None)
            route_j = next((r for r in routes if j in r), None)
            
            # 다른 경로에 있고, 각각 끝점이며, 용량을 초과하지 않는지 확인
            if route_i and route_j and route_i != route_j:
                is_i_end = (route_i[0] == i or route_i[-1] == i)
                is_j_end = (route_j[0] == j or route_j[-1] == j)
                
                if is_i_end and is_j_end:
                    total_demand = sum(parser.nodes[n]['demand'] for n in route_i + route_j)
                    if total_demand <= max_capacity:
                        available_merges.append((s, i, j, route_i, route_j))
        
        if not available_merges:
            break
            
        # Savings가 높은 순으로 정렬
        available_merges.sort(key=lambda x: x[0], reverse=True)
        
        # [무작위성 부여] 상위 top_k개 중 하나를 무작위로 선택
        # 후보가 top_k보다 적으면 전체 중 선택
        current_top_k = min(len(available_merges), top_k)
        selected_idx = random.randint(0, current_top_k - 1)
        s, i, j, route_i, route_j = available_merges[selected_idx]
        
        # 방향 맞춰서 병합
        if route_i[-1] == i and route_j[0] == j:
            new_route = route_i + route_j
        elif route_i[0] == i and route_j[-1] == j:
            new_route = route_j + route_i
        elif route_i[-1] == i and route_j[-1] == j:
            new_route = route_i + route_j[::-1]
        else:
            new_route = route_i[::-1] + route_j
        
        routes.remove(route_i)
        routes.remove(route_j)
        routes.append(new_route)

    # 4. Two-opt 적용 (각 경로 최적화)
    optimized_routes = []
    for route in routes:
        if len(route) < 3:
            optimized_routes.append(route)
            continue
            
        best_route = route
        improved = True
        while improved:
            improved = False
            for i in range(len(best_route) - 1):
                for j in range(i + 2, len(best_route) + 1):
                    new_route = best_route[:i] + best_route[i:j][::-1] + best_route[j:]
                    if get_route_dist(new_route, parser) < get_route_dist(best_route, parser):
                        best_route = new_route
                        improved = True
        optimized_routes.append(best_route)

    return optimized_routes

def get_route_dist(route, parser):
    depot = parser.nodes[0]
    if not route: return 0
    dist = calculate_distance(depot, parser.nodes[route[0]]) # 데포 -> 첫 고객
    for i in range(len(route) - 1):
        dist += calculate_distance(parser.nodes[route[i]], parser.nodes[route[i+1]])
    dist += calculate_distance(parser.nodes[route[-1]], depot) # 마지막 고객 -> 데포
    return distgit