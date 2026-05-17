import json
import random
from cvrp_utils import CVRPParser

# 데이터 파싱
parser = CVRPParser(r'C:\Users\USER\Desktop\SDO\GOPT_cvrp\3L_CVRP\3l_cvrp01.txt')
capacity_limit = parser.vehicle_info['capacity']
customer_ids = list(range(1, parser.num_customers + 1))

valid_single_routes = []

# 단일 차량 기준 500개의 경로 생성
while len(valid_single_routes) < 500:
    random.shuffle(customer_ids)
    current_route = []
    current_load = 0
    
    for customer in customer_ids:
        demand = parser.nodes[customer]['demand']
        if current_load + demand <= capacity_limit:
            current_route.append(customer)
            current_load += demand
        else:
            break  # 용량 초과 시 현재까지의 경로만 저장
            
    # 중복되지 않는 경로만 추가
    if current_route not in valid_single_routes:
        valid_single_routes.append(current_route)

# json 파일로 저장 (메모장으로 확인 가능)
with open('eval_routes_500.json', 'w', encoding='utf-8') as f:
    json.dump(valid_single_routes, f, indent=4)

print("500개의 경로가 eval_routes_500.json 파일에 저장되었습니다.")