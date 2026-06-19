from src.agents.critic_agent import CriticGatingAgent
from src.agents.saber_agent import SABERGatingAgent
from src.agents.saber_paper_agent import SaberPaperAgent
from src.agents.saber_block_agent import SaberBlockAgent
from src.agents.oracle_agent import OracleGatingAgent
from src.agents.reflexion_agent import ReflexionAgent
from src.agents.vanilla import VanillaAgent


def get_agent(method: str, **kwargs):
    """
    Factory. method ∈ {"ours", "saber", "saber_old", "oracle", "reflexion", "vanilla"}.

    "saber"     = faithful SABER (Cuadron et al. 2025): mutation-gated USER
                  verification + targeted reflection (SaberPaperAgent).
    "saber_old" = the original simplified baseline (autonomous 4-way LLM critic,
                  never reaches the user) — kept only for reference.
    kwargs are forwarded to the agent constructor.
    """
    dispatch = {
        "ours": CriticGatingAgent,
        "saber": SaberPaperAgent,        # faithful SABER (mechanisms 1+2)
        "saber_block": SaberBlockAgent,  # full SABER (1+2+3 block-based context cleaning)
        "saber_old": SABERGatingAgent,   # original wrong/simplified version
        "oracle": OracleGatingAgent,
        "reflexion": ReflexionAgent,
        "vanilla": VanillaAgent,
    }
    if method not in dispatch:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(dispatch)}")
    return dispatch[method](**kwargs)
