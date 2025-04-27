from classes.servitor import Servitor, Brain


def main(config_path):
    brain = Brain("data/llm/mistral-7b-instruct-v0.1.Q4_K_M.gguf")
    servitor = Servitor(config_path, brain)
    servitor


if __name__ == "__main__":
    main("config/config.yaml")
