from pydantic import BaseModel, Field, create_model
from ollama import chat
from enum import Enum
from typing import Union, Literal
import json

MODEL = "llama3.2:1b"


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
    known_items: list[Item] = []
    known_locations: list[str] = []
    known_agents: list[str] = []
    current_location: str
    motivation: Union[SeekMotivation, BeHelpfulMotivation]
    knowledge: list[Knowledge]


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

    agents = [seeker_agent, knower_agent]

    past_actions = []

    while True:
        for agent in agents:
            prompt = f"""Here is your state reflected in JSON:
{agent.model_dump_json()}
Here are the past history of actions in this world:
{json.dumps(past_actions)}

Respond in plain text with what you'd like to do next."""
            print(prompt)
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

            past_actions.append({agent.name: response["message"]["content"]})
            print(past_actions[-1])


if __name__ == "__main__":
    main()
