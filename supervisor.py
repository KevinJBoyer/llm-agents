from pydantic import BaseModel, Field, create_model
from ollama import chat
from enum import Enum
from typing import Union, Literal
import json

MODEL = "llama3.1:70b"


class Sword(BaseModel):
    type: Literal["Sword"] = "Sword"


class Stone(BaseModel):
    type: Literal["Stone"] = "Stone"


class Item(BaseModel):
    item: Union[Sword, Stone] = Field(..., discriminator="type")


class Location(BaseModel):
    of: Item
    location: str


class Knowledge(BaseModel):
    knowledge: Location


class Information(BaseModel):
    information: set[Knowledge]


class SeekMotivation(BaseModel):
    motivation_name: str = "Seek item"
    item: Item


class BeHelpfulMotivation(BaseModel):
    motivation_name: str = "Be helpful"
    pass


class Agent(BaseModel):
    name: str
    inventory: list[Item] = []
    known_items: list[Item] = []
    known_locations: set[str] = set()
    known_agents: list[str] = []
    current_location: str
    motivation: Union[SeekMotivation, BeHelpfulMotivation]
    knowledge: list[Knowledge]

    def do_next_action(self, global_past_actions):
        print("")
        print(f"It's {self.name}'s turn! ===== ")
        prompt = f"""Here is your state reflected in JSON:
        {self.model_dump_json()}
        Here are the past history of actions in this world:
        {json.dumps(global_past_actions)}
        
        Respond in plain text with what you'd like to do next."""
        response = chat(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=MODEL,
            # stream=True,
        )
        global_past_actions.append({self.name: response["message"]["content"]})

        print(f"{self.name} has decided to:")
        print(response["message"]["content"])

        print("")
        print(f"{self.name}'s state")
        print(self.model_dump_json())

        return global_past_actions[-1]


def tool_grant_sword(agent):
    agent.known_items.append(Item(item=Sword()))

def tool_move(agent, location):
    agent.current_location = location
    agent.known_locations.add(location)

def tool_stand_by_and_do_nothing():
    pass

TOOL_FUNCTIONS = {
    "tool_grant_sword": tool_grant_sword,
    "tool_move": tool_move,
    "tool_stand_by_and_do_nothing": tool_stand_by_and_do_nothing
}

class Supervisor:
    def update_agent_state(self, agent, requested_action):
        print("")
        print(f"Supervisor responding to {agent.name}'s request ===== ")
        response = chat(
            messages=[
                {
                    "role": "user",
                    "content": f"""
                    You are responsible for updating the state of each agent
                    in a game consistent with reasonable and responsible game rules.
                    
                    Each tool should only be called if you have decided that the
                    agent has fulfilled the conditions for the state update to be
                    performed.
                    
                    This agent's current state is reflected in JSON:
                    
                    {agent.model_dump_json()}
                    
                    This is what the agent has requested to do:
                    {requested_action}
                    
                    Remember to be reasonable and sensible!!                    
                    """,
                }
            ],
            model=MODEL,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "tool_grant_sword",
                        "description": "Grant the agent a sword",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "tool_move",
                        "description": "Move the agent to a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                            },
                            "required": ["location"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                },

                {
                    "type": "function",
                    "function": {
                        "name": "tool_stand_by_and_do_nothing",
                        "description": "No action required",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                }
            ],
            # stream=True,
        )

        print(response["message"]["content"])
        print(response["message"]["tool_calls"])

        # grab the tool calls from the ressponse and call them
        response_tool_calls = response["message"]["tool_calls"]
        for tool_call in response_tool_calls:
            fn_name = tool_call["function"]["name"]
            match fn_name:
                case "tool_move":
                    TOOL_FUNCTIONS[fn_name](
                        agent, tool_call["function"]["arguments"]["location"]
                    )
                case "tool_grant_sword":
                    TOOL_FUNCTIONS[fn_name](agent)
                case "tool_stand_by_and_do_nothing":
                    TOOL_FUNCTIONS[fn_name]()

def main():
    seeker_agent = Agent(
        name="Seeker",
        current_location="city",
        known_items=[Item(item=Stone()), Item(item=Sword())],
        known_locations=["city"],
        known_agents=["Knower"],
        motivation=SeekMotivation(item=Item(item=Sword())),
        knowledge=[],
    )
    knower_agent = Agent(
        name="Knower",
        current_location="city",
        known_locations=["city", "field"],
        known_agents=["Seeker"],
        motivation=BeHelpfulMotivation(),
        knowledge=[
            Knowledge(knowledge=Location(of=Item(item=Sword()), location="field"))
        ],
    )

    supervisor = Supervisor()

    agents = [seeker_agent, knower_agent]
    global_past_actions = []

    while True:
        for agent in agents:
            requested_action = agent.do_next_action(global_past_actions)
            supervisor.update_agent_state(agent, requested_action)


if __name__ == "__main__":
    main()
