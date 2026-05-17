import os
import argparse
from omegaconf import OmegaConf

# 현재 파일(arguments.py)이 있는 경로 구하기
curr_path = os.path.dirname(os.path.abspath(__file__))

def get_args():
    parser = argparse.ArgumentParser()

    # CVRP 데이터 폴더 경로 (단일 파일이 아닌 폴더)
    parser.add_argument('--data-dir', type=str, default="3L_CVRP", 
                        help="Directory containing CVRP text files")

    # [변경] 기본 설정 파일을 cvrp_config.yaml로 변경
    parser.add_argument('--config', type=str, default="cfg/cvrp_config.yaml",
                        help="Path to the configuration file")
    
    parser.add_argument('--ckp', type=str, default=None, 
                        help="Path to the model to be tested")
    parser.add_argument('--no-cuda', action='store_true',
                        help='Cuda will be enabled by default')
    parser.add_argument('--device', type=int, default=0, 
                        help='Which GPU will be called')
    parser.add_argument('--test-episode', type=int, default=1000, 
                        help='Number of episodes for evaluation')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment while testing')
    
    args, unknown = parser.parse_known_args()

    if not os.path.isabs(args.config):
        args.config = os.path.normpath(os.path.join(curr_path, args.config))

    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.normpath(os.path.join(curr_path, args.data_dir))

    if args.ckp and not os.path.isabs(args.ckp):
        args.ckp = os.path.normpath(os.path.join(curr_path, args.ckp))
        
    cfg = OmegaConf.load(args.config)
    
    # -----------------------------------------------------------------
    # 아래 로직은 YAML에 적힌 '기본값'을 기준으로 박스 크기 범위를 계산합니다.
    # -----------------------------------------------------------------
    max_dim = max(cfg.env.container_size)
    box_small = int(max_dim / 10)
    box_big = int(max_dim / 2)
    
    # box_range = (5, 5, 5, 25, 25, 25)
    box_range = (box_small, box_small, box_small, box_big, box_big, box_big)

    if cfg.get("env.step") is not None:
        step = cfg.env.step
    else:
        step = box_small

    # 박스 사이즈 셋 생성 (RandomBoxCreator용, CVRP 모드에서는 참조용)
    box_size_set = []
    # range step이 0이면 에러나므로 최소 1로 보정
    if step < 1: step = 1
    
    for i in range(box_range[0], box_range[3] + 1, step):
        for j in range(box_range[1], box_range[4] + 1, step):
            for k in range(box_range[2], box_range[5] + 1, step):
                box_size_set.append((i, j, k))
    
    cfg.env.box_small = box_small
    cfg.env.box_big = box_big
    cfg.env.box_size_set = box_size_set
    
    # CUDA 설정
    cfg.cuda = not args.no_cuda 

    # 커맨드라인 인자(args)로 Config(cfg) 덮어쓰기
    # 이 과정에서 --data-path 값이 cfg에 병합됩니다.
    cfg = OmegaConf.merge(cfg, vars(args))

    return cfg


if __name__ == "__main__":
    args = get_args()
    print(f"Data Dir: {args.data_dir}")
    print(f"Config: {args.config}")