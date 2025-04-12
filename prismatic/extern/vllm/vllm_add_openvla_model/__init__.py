from vllm import ModelRegistry


def register():
    from .openvla import OpenVLAForActionPrediction
    if "OpenVLAForActionPrediction" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("OpenVLAForActionPrediction", OpenVLAForActionPrediction)
