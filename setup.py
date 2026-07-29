"""
setup.py — editable install for the Identity Document Intelligence System.
"""
from setuptools import setup, find_packages

setup(
    name="identity-doc-intelligence",
    version="1.0.0",
    description="End-to-end VLM pipeline for identity document field extraction and forgery detection",
    packages=find_packages(exclude=["tests*", "notebooks*", "scripts*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "trl>=0.8.6",
        "Pillow>=10.0.0",
        "opencv-python>=4.9.0",
        "Faker>=24.0.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "scikit-learn>=1.4.0",
        "wandb>=0.16.0",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.0",
    ],
    entry_points={
        "console_scripts": [
            "idis-generate=stage1_data_factory.generator:main",
            "idis-split=stage1_data_factory.split:main",
            "idis-benchmark=stage2_ocr_benchmark.benchmark:main",
            "idis-trocr-finetune=stage2_ocr_benchmark.finetune_trocr:main",
            "idis-sft=stage3_vlm_sft.train:main",
            "idis-infer=stage3_vlm_sft.inference:main",
            "idis-forgery-train=stage4_forgery_detection.train:main",
            "idis-forgery-eval=stage4_forgery_detection.evaluate:main",
            "idis-dpo=stage5_dpo.train:main",
            "idis-dpo-eval=stage5_dpo.evaluate:main",
            "idis-harness=stage6_eval_harness.harness:main",
            "idis-report=stage6_eval_harness.report:main",
            "idis-upload=stage6_eval_harness.hub_upload:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
    ],
)
