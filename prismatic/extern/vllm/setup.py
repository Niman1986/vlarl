from setuptools import setup

setup(name='vllm_add_openvla_model',
      version='0.1',
      packages=['vllm_add_openvla_model'],
      entry_points={
          'vllm.general_plugins':
          ["register_openvla_model = vllm_add_openvla_model:register"]
      })
