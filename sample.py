from omegaconf import DictConfig

import hydra
@hydra.main(config_path=".",config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())


if __name__ == '__main__':
    main()