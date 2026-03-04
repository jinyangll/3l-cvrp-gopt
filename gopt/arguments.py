import os
import argparse
from omegaconf import OmegaConf

# 현재 파일(arguments.py)이 있는 경로 구하기
curr_path = os.path.dirname(os.path.abspath(__file__))

def get_args():
    parser = argparse.ArgumentParser()
    
    #CVRP 데이터 파일 경로
    parser.add_argument('--data-path', type=str, default="3L_CVRP/3l_cvrp01.txt", 
                        help="Path to the CVRP text file containing vehicle and item info")

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

    # Config 파일 로드 로직
    try:
        # config 경로가 절대 경로가 아니면 현재 파일 기준으로 찾음
        if not os.path.isabs(args.config):
            args.config = os.path.join(curr_path, args.config)
            
        cfg = OmegaConf.load(args.config)
    except FileNotFoundError:
        print(f"No configuration file found at: {args.config}")
        # 파일이 없으면 빈 설정으로라도 진행하거나 종료 (여기서는 진행 시도)
        exit()
    
    # -----------------------------------------------------------------
    # 아래 로직은 YAML에 적힌 '기본값'을 기준으로 박스 크기 범위를 계산합니다.
    # 하지만 ts_train.py가 실행되면 텍스트 파일을 읽고 이 값들을 다시 덮어쓰게 됩니다.
    # 초기화를 위해 남겨두는 코드입니다.
    # -----------------------------------------------------------------
    
    # 컨테이너 크기 기반으로 box_small/big 계산 (YAML 기본값 기준)
    # 실제 학습 시에는 ts_train.py에서 파일 데이터에 맞게 재조정됩니다.
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
    print(f"Data Path: {args.data_path}")
    print(f"Config: {args.config}")