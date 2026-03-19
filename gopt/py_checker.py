import os
import sys
import torch
import gymnasium as gym
from tianshou.data import Batch

from ts_train import build_net
import arguments
from tools import set_seed, CategoricalMasked, registration_envs
from masked_ppo import MaskedPPOPolicy
from cvrp_utils import CVRPParser
from envs.Packing.binCreator import EvalBoxCreator
# from binCreator import EvalBoxCreator

args = None
test_env = None
policy = None
parser = None

def initialize():
    global args, test_env, policy, parser, device

    # 커스텀 환경 일괄 등록
    registration_envs()

    #전역 초기화 (C++에서 import 시 1회만 실행)
    args = arguments.get_args()
    args.train.algo = args.train.algo.upper()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    set_seed(args.seed, args.cuda, args.cuda_deterministic)

    # 경로 설정
    # cvrp_file = r"C:\Users\USER\Desktop\SDO\GOPT_cvrp\3L_CVRP\3l_cvrp01.txt"
    # args.ckp = r"C:\Users\USER\Desktop\SDO\GOPT_cvrp\learned_model\policy_step_best7.pth"

    # cvrp_file = "3L_CVRP/3l_cvrp01.txt"
    # args.ckp = "learned_model/policy_step_best7.pth"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    cvrp_file = os.path.join(current_dir, "3L_CVRP", "3l_cvrp01.txt")
    args.ckp = os.path.join(current_dir, "learned_model", "policy_step_best7.pth")

    parser = CVRPParser(cvrp_file)
    veh_info = parser.vehicle_info
    args.env.container_size = (veh_info['length'], veh_info['width'], veh_info['height'])

    # 환경 생성
    test_env = gym.make(
        args.env.id, 
        container_size=args.env.container_size,
        enable_rotation=args.env.rot,
        data_type="cvrp",
        item_set=None,
        reward_type=args.train.reward_type,
        action_scheme=args.env.scheme,
        k_placement=args.env.k_placement,
        is_render=False, 
        cvrp_parsers=parser
    )

    # 생성기를 EvalBoxCreator로 교체
    test_env.unwrapped.box_creator = EvalBoxCreator(parser)

    # 모델 구성 및 로드
    actor, critic = build_net(args, device)
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.opt.lr)

    policy = MaskedPPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=CategoricalMasked,
        discount_factor=args.train.gamma,
        eps_clip=args.train.clip_param,
        action_space=test_env.action_space,
    )

    policy.load_state_dict(torch.load(args.ckp, map_location=device))
    policy.eval()



#C++ 호출용 check 함수
def check(route):

    global test_env, policy

    # test_env == None이면 initialize()
    if test_env is None:
        initialize()
    
    # initialize() 후에도 None이면 error
    if test_env is None:
        print("Error: test_env is still None after initialization!")
        return False



    # C++에서 전달받은 route 주입
    test_env.unwrapped.box_creator.set_route(route)
    
    # 주입된 route로 환경 리셋
    obs, info = test_env.reset()
    done = False
    
    # 에피소드 실행
    while not done:
        batch = Batch(obs=[obs], info=[info])
        
        with torch.no_grad():
            result = policy(batch)
            
        action = result.act[0].item()
        
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        
    return info.get('is_success', False)
