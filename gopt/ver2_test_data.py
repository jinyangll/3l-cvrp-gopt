import os
import sys
import time
import json
import csv
import torch
import gymnasium as gym
from tianshou.data import Batch

# 루트 디렉토리 설정
curr_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curr_path)

# 기존 모듈 임포트
from cvrp_utils import CVRPParser
# 💡 수정된 부분: 파일 구조에 맞게 폴더 경로(envs.Packing) 명시
from envs.Packing.binCreator import EvalBoxCreator
from ts_train import build_net
import arguments
from tools import set_seed, registration_envs, CategoricalMasked
from masked_ppo import MaskedPPOPolicy

def evaluate_500():
    args = arguments.get_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    set_seed(args.seed, args.cuda, args.cuda_deterministic)

    # 1. 500개 평가 경로 로드
    with open('eval_routes_500.json', 'r', encoding='utf-8') as f:
        routes = json.load(f)

    # 2. 파서 세팅
    cvrp_file = r"C:\Users\USER\Desktop\SDO\GOPT_cvrp\3L_CVRP\3l_cvrp01.txt"
    parser = CVRPParser(cvrp_file)
    
    veh_info = parser.vehicle_info
    args.env.container_size = (veh_info['length'], veh_info['width'], veh_info['height'])

    # 3. 환경 등록 및 초기화
    registration_envs()
    env = gym.make(
        args.env.id, 
        container_size=args.env.container_size,
        enable_rotation=args.env.rot,
        data_type="cvrp",          
        reward_type=args.train.reward_type,
        action_scheme=args.env.scheme,
        k_placement=args.env.k_placement,
        is_render=False,
        cvrp_parsers=parser         
    )
    
    # 평가용 BoxCreator 주입
    env.unwrapped.box_creator = EvalBoxCreator(parser)

    # 4. 모델 로드
    actor, critic = build_net(args, device)
    optim = torch.optim.Adam(actor.parameters(), lr=args.opt.lr)
    
    policy = MaskedPPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=CategoricalMasked,
        action_space=env.action_space,
    )
    
    ckp_path = r"C:\Users\USER\Desktop\SDO\GOPT_cvrp\learned_model\ver2\policy_step_best_0310.pth"
    policy.load_state_dict(torch.load(ckp_path, map_location=device))
    policy.eval()

    results = []

    print(f"총 {len(routes)}개의 경로 평가 시작...")

    # 5. 순수 while문을 통한 속도 측정 및 평가
    for idx, route in enumerate(routes):
        env.unwrapped.box_creator.set_route(route)
        obs, info = env.reset()
        
        done = False
        start_time = time.time()
        
        while not done:
            batch = Batch(obs=[obs], info=info)
            with torch.no_grad():
                act = policy(batch).act[0]
            
            obs, reward, terminated, truncated, info = env.step(act)
            done = terminated or truncated
            
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        is_success = info.get('is_success', False)
        packed_items = info.get('counter', 0)
        total_items = info.get('total_items', 0)
        ratio = info.get('ratio', 0.0)
        
        results.append([idx + 1, str(route), is_success, packed_items, total_items, ratio, elapsed_time])
        print(f"[{idx+1}/500] 성공: {is_success} | 시간: {elapsed_time:.4f}s | 공간활용: {ratio:.4f} | 패킹: {packed_items}/{total_items}")

    # 6. CSV 파일 저장
    csv_filename = 'ver2_evaluation_results_500.csv'
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Route", "Is_Success", "Packed_Items", "Total_Items", "Space_Ratio", "Time(s)"])
        writer.writerows(results)

    print(f"\n모든 평가가 완료되었습니다! 결과가 '{csv_filename}' 에 저장되었습니다.")

if __name__ == '__main__':
    evaluate_500()