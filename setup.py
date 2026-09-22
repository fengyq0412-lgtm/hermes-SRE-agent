from setuptools import find_packages, setup


setup(
    name="hermes-sre-agent",
    version="0.1.0",
    description="证据优先、审批受控的 SRE 故障诊断智能体",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"hermes_sre_agent": ["web_assets/*"]},
    entry_points={"console_scripts": [
        "hermes=hermes_sre_agent.cli:main",
        "hermes-web=hermes_sre_agent.web_server:main",
    ]},
)
