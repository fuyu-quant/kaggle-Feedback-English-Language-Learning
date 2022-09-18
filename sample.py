

import hydra
@hydra.main(config_name="config")

def main(cfg):
    print(cfg.seed)

if __name__ == '__main__':
    main()