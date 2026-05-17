import os
import sys
curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(curr_path) 
sys.path.append(parent_path) 

import random 
import gymnasium as gym
import torch
from tianshou.utils.net.common import ActorCritic

from ts_train import build_net
import arguments
from tools import *
from mycollector import PackCollector
from masked_ppo import MaskedPPOPolicy

# [추가] CVRP 데이터 처리를 위한 import
from cvrp_utils import CVRPParser 


def test(args):

    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda", args.device)
    else:
        device = torch.device("cpu")
        
    set_seed(args.seed, args.cuda, args.cuda_deterministic)

    # CVRP 데이터 파싱 및 환경 설정 (train.py와 동일하게 맞춤)
    # 주의: 실행 시 args.data_path가 올바른 데이터 파일을 가리키고 있어야 합니다.
    #cvrp_file = args.data_path 
    cvrp_file = r"C:\Users\USER\Desktop\SDO\GOPT_cvrp\3L_CVRP\3l_cvrp13.txt"
    if not os.path.exists(cvrp_file):
        print(f"Error: Data file not found at {cvrp_file}")
        print("Please check 'args.data_path' in arguments.py or pass it via command line.")
        exit()
        
    print(f"Loading CVRP Data from: {cvrp_file}")
    parser = CVRPParser(cvrp_file)
    
    # 차량 정보로 컨테이너 크기 업데이트 (학습 환경과 일치시키기 위함)
    veh_info = parser.vehicle_info
    real_container_size = (veh_info['length'], veh_info['width'], veh_info['height'])
    args.env.container_size = real_container_size
    print(f"Auto-configured Container Size: {args.env.container_size}")

    # 환경 생성 (parser 전달 및 data_type='cvrp' 고정)
    test_env = gym.make(
        args.env.id, 
        container_size=args.env.container_size,
        enable_rotation=args.env.rot,
        data_type="cvrp",          # CVRP 모드 강제
        item_set=None,             # CVRP 모드에서는 parser가 아이템을 공급하므로 None
        reward_type=args.train.reward_type,
        action_scheme=args.env.scheme,
        k_placement=args.env.k_placement,
        is_render=args.render,     # 렌더링 여부
        cvrp_parsers=[parser]         # 파서 객체 전달
    )

    # network
    actor, critic = build_net(args, device)
    actor_critic = ActorCritic(actor, critic)

    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.opt.lr, eps=args.opt.eps)
    
    # RL agent 
    dist = CategoricalMasked

    policy = MaskedPPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        discount_factor=args.train.gamma,
        eps_clip=args.train.clip_param,
        advantage_normalization=False,
        vf_coef=args.loss.value,
        ent_coef=args.loss.entropy,
        gae_lambda=args.train.gae_lambda,
        action_space=test_env.action_space,
    )
    
    policy.eval()
    
    # 모델 로드 (try-except 제거)
    print(f"Loading model from: {args.ckp}")
    if not os.path.exists(args.ckp):
        print(f"No model found at {args.ckp}")
        exit()
        
    policy.load_state_dict(torch.load(args.ckp, map_location=device))
    print("Model loaded successfully.")

    test_collector = PackCollector(policy, test_env)

    # Evaluation
    print(f"Start testing (Render: {args.render})...")
    result = test_collector.collect(n_episode=args.test_episode, render=args.render)
    
    for i in range(args.test_episode):
        # result['nums']가 에피소드 별 결과를 담고 있다고 가정
        ratio = result['ratios'][i] if 'ratios' in result else result['ratio']
        num = result['nums'][i] if 'nums' in result else result['num']
        print(f"episode {i+1}\t => \tratio: {ratio:.4f} \t| items packed: {num}")
        
    print('All cases have been done!')
    print('----------------------------------------------')
    print('average space utilization: %.4f'%(result['ratio']))
    print('average put item number: %.4f'%(result['num']))
    print("standard variance: %.4f"%(result['ratio_std']))


if __name__ == '__main__':
    registration_envs()
    args = arguments.get_args()
    args.train.algo = args.train.algo.upper()
    args.train.step_per_collect = args.train.num_processes * args.train.num_steps  

    #모델 경로
    args.ckp = r"C:\Users\USER\Desktop\SDO\GOPT_cvrp\learned_model\ver2\policy_step_best_0333"
    args.render = True
    #테스트 에피소드 수
    args.test_episode = 1
    #시드 설정
    import time
    args.seed = int(time.time()) 
    #args.seed = 1770624124
    #5번 파일(1번 인스턴스, 지지제약 50%)
    #그냥 잘채우는거 : 1770623879, 1770613155, 1770610458, 1770624124
    #제약이 보이는거 : 1770610558, 1770625166

    print(f"Current Seed: {args.seed}")

    # CVRP 데이터 파일 경로 설정
    # 학습할 때 사용했던 데이터 파일 경로가 args.data_path에 들어가야함.
    # 만약 arguments.py 기본값이 다르다면 아래 주석을 풀고 경로를 지정
    # args.data_path = "/path/to/your/cvrp_data.txt" 

    print(f"Testing with container dimension: {args.env.container_size} (will be auto-updated)")
    test(args)