from omegaconf import DictConfig

import hydra
@hydra.main(config_path=".",config_name="config.yaml",version_base=None)
def main(cfg: DictConfig) -> None:
    #print(cfg.pretty())
    print(cfg.setting.seed)


if __name__ == '__main__':
    main()