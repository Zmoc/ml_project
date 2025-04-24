from classes.servitor import Servitor


def main(config_path):
    servitor = Servitor(config_path)
    servitor


if __name__ == "__main__":
    main("config/config.yaml")
