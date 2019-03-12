class GeneratorBase(object):
    def __init__(self):
        pass

    def encode(self, message: str, seed_text = " ".join(["word"] * 100)) -> str:
        raise NotImplementedError("Has to be overriden!")