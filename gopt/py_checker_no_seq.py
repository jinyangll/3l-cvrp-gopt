import os
import sys
import torch
import gymnasium as gym
from tianshou.data import Batch
import itertools

from ts_train import build_net
import arguments
from tools import set_seed, CategoricalMasked, registration_envs
from masked_ppo import MaskedPPOPolicy
from cvrp_utils import CVRPParser

# 💡 수정된 부분: envs.Packing 경로 명시
from envs.Packing.binCreator import EvalBoxCreator


args = None
test_env = None
policy = None
parser = None

def initialize():
    global args, test_env, policy, parser, device


    # 커스텀 환경 일괄 등록 및 전역 초기화 (임포트 시 즉시 실행)
    registration_envs()

    args = arguments.get_args()
    args.train.algo = args.train.algo.upper()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    set_seed(args.seed, args.cuda, args.cuda_deterministic)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    cvrp_file = os.path.join(BASE_DIR, "3L_CVRP", "3l_cvrp02.txt")
    args.ckp = os.path.join(BASE_DIR, "learned_model", "ver2", "policy_step_best_0326.pth")

    parser = CVRPParser(cvrp_file)
    veh_info = parser.vehicle_info
    args.env.container_size = (veh_info["length"], veh_info["width"], veh_info["height"])

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
        cvrp_parsers=parser,
    )

    # 생성기를 EvalBoxCreator로 교체
    test_env.unwrapped.box_creator = EvalBoxCreator(parser)

    # 모델 구성 및 로드
    actor, critic = build_net(args, device)
    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=args.opt.lr
    )

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


# C++ 호출용 check 함수
def check_no_seq(route):
    global test_env, policy

    # test_env == None이면 initialize()
    if test_env is None:
        initialize()
    
    # initialize() 후에도 None이면 error
    if test_env is None:
        print("Error: test_env is still None after initialization!")
        return False

    # 입력받은 route의 모든 가능한 순열 조합 생성 (n!)
    possible_routes = list(itertools.permutations(route))

    # 각 순열 조합에 대해 Feasible 여부 검사
    for perm_route in possible_routes:
        current_perm = list(perm_route)

        # 순열 조합 주입 및 리셋
        test_env.unwrapped.box_creator.set_route(current_perm)
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

        # 단 1개의 조합이라도 성공하면 즉시 True 반환
        if info.get("is_success", False):
            return True

    return False
