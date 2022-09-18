from omegaconf import DictConfig, OmegaConf

import hydra
@hydra.main(config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == '__main__':
    main()