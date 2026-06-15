from src.agents.critic_agent import CriticGatingAgent
from src.agents.saber_agent import SABERGatingAgent
from src.agents.oracle_agent import OracleGatingAgent
from src.agents.reflexion_agent import ReflexionAgent
from src.agents.vanilla import VanillaAgent


def get_agent(method: str, **kwargs):
    """
    Factory. method ∈ {"ours", "saber", "oracle", "reflexion", "vanilla"}.
    kwargs are forwarded to the agent constructor.
    """
    dispatch = {
        "ours": CriticGatingAgent,
        "saber": SABERGatingAgent,
        "oracle": OracleGatingAgent,
        "reflexion": ReflexionAgent,
        "vanilla": VanillaAgent,
    }
    if method not in dispatch:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(dispatch)}")
    return dispatch[method](**kwargs)
