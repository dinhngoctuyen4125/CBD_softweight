from setuptools import setup, find_packages

setup(
    name='uld',
    version='1.0',
    packages=find_packages(),
    # Phiên bản được ghim trong environment.yaml (conda). Ở đây chỉ khai các gói
    # thực sự được import, không đọc requirements.txt — file đó đã bị xoá khỏi repo.
    install_requires=[
        'torch',
        'transformers',
        'peft',
        'datasets',
        'accelerate',
        'hydra-core',
        'omegaconf',
        'safetensors',
        'structlog',
        'codetiming',
        'lightning',
        'pytorch-lightning',
        'pandas',
        'numpy',
        'matplotlib',
    ],
    description='CBD-DFB: discriminative-subspace unlearning for deprecated-API detection',
    license='MIT',
)
