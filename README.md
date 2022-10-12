
<p align="center">
<h1 align="center"> Multi-Agent Distributional Reward Estimation (DRE-MARL) </h1>
</p>



Code for "Distributional Reward Estimation for Effective Multi-Agent Deep Reinforcement Learning".

![Framework](Framework.png)

First clone the code and installation of the relevant package.

    git clone https://github.com/JF-Hu/DRE-MARL.git
    cd DRE-MARL
    pip install -r requirements.txt

Before you start we strongly recommend that you register a `wandb` account.
This will record graphs and curves during the experiment.
If you want, complete the login operation in your shell. Enter the following command and follow the prompts to complete the login.

    wandb login

API keys can be found in User Settings page https://wandb.ai/settings. For more information you can refer to https://docs.wandb.ai/quickstart .

Next is how to replicat all experiments:
## For Model Training
### Run MARL on Cooperative Navigation (CN)

If use default training config:

    python train_CN.py 

Other settings can be found in `algorithm/hyperpara_setting.py`

### Run MARL on Reference (REF)

If use default training config:

    python train_REF.py 

Other settings can be found in `algorithm/hyperpara_setting.py`

### Run MARL on Treasure Collection (TREA)

If use default training config:

    python train_TREA.py 

Other settings can be found in `algorithm/hyperpara_setting.py` 
