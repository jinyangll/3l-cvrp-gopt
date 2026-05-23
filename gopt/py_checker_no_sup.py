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

args = None
test_env = None
policy = None
parser = None


def initialize():
  global args, test_env, policy, parser, device


  registration_envs()

  args = arguments.get_args()
  args.train.algo = args.train.algo.upper()

  # No Support 환경 ID로 강제 고정
  args.env.id = 'OnlinePack-v1-NoSup'

  device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
  set_seed(args.seed, args.cuda, args.cuda_deterministic)

  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  cvrp_file = os.path.join(BASE_DIR, "3L_CVRP", "3l_cvrp02.txt")
  args.ckp = os.path.join(BASE_DIR, "learned_model", "ver2", "policy_step_best_0326.pth")

  parser = CVRPParser(cvrp_file)
  veh_info = parser.vehicle_info
  args.env.container_size = (veh_info['length'], veh_info['width'], veh_info['height'])

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

  test_env.unwrapped.box_creator = EvalBoxCreator(parser)

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



def check_no_sup(route):

    global test_env, policy

    # test_env == None이면 initialize()
    if test_env is None:
        initialize()
    
    # initialize() 후에도 None이면 error
    if test_env is None:
        print("Error: test_env is still None after initialization!")
        return False


    test_env.unwrapped.box_creator.set_route(route)
    obs, info = test_env.reset()
    done = False
    
    while not done:
        batch = Batch(obs=[obs], info=[info])
        with torch.no_grad():
            result = policy(batch)
            
        action = result.act[0].item()
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        
    return info.get('is_success', False)

'''
if __name__ == "__main__":
    # 테스트해볼 임의의 노드 방문 순서 (경로)
    test_route = [1, 2, 3, 4] 
    
    print(f"경로 테스트 시작: {test_route}")
    is_feasible = check(test_route)
    
    if is_feasible:
        print("결과: 적재 가능 (Feasible)")
    else:
        print("결과: 적재 불가능 (Infeasible)")
'''