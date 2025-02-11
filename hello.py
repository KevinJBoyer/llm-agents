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

    def produce_next_action(self, available_actions: list[any]):
        response = chat(
            messages=[
                {
                    "role": "user",
                    "content": f"Choose an action based on your current state, which is {self.model_dump_json()}",
                }
            ],
            model=MODEL,
            format=available_actions.model_json_schema(),
            # stream=True,
        )

        return response["message"]["content"]

    def receive_action(self):
        # should update state
        pass


def possible_next_actions(agent: Agent):
    available_agents_to_ask = Enum(
        "available_agents_to_ask", {l.upper(): l for l in agent.known_agents}
    )

    valid_location_knowledge = Enum(
        "valid_location_knowledge",
        {l.item.type: l.item.type for l in agent.known_items},
    )

    valid_ask_actions = create_model(
        "valid_ask_actions",
        type=(Literal["ask_action"], "ask_action"),
        agent_to_ask=(available_agents_to_ask, ...),
        question_to_gain_knowledge_desired=(str, ...),
        knowledge_desired=(valid_location_knowledge, ...),
    )

    available_travel_locations = Enum(
        "available_travel_locations",
        {l.upper(): l for l in agent.known_locations if l != agent.current_location},
    )
    valid_travel_actions = create_model(
        "valid_travel_actions",
        type=(Literal["travel_action"], "travel_action"),
        location=(available_travel_locations, ...),
    )

    if len(agent.known_locations) > 1:
        available_actions = Union[valid_travel_actions, valid_ask_actions]
    else:
        available_actions = valid_ask_actions

    print(available_actions.model_json_schema())
    return create_model("agent_actions", action=(available_actions, ...))


def get_available_actions(agent: Agent):
    available_agents_to_ask = Enum(
        "available_agents_to_ask", {l.upper(): l for l in agent.known_agents}
    )

    valid_location_knowledge = Enum(
        "valid_location_knowledge",
        {item.item.type: item.item.type for item in agent.known_items},
    )

    valid_ask_actions = create_model(
        "valid_ask_actions",
        type=(Literal["ask_action"], "ask_action"),
        agent_to_ask=(available_agents_to_ask, ...),
        question_to_gain_knowledge_desired=(str, ...),
        knowledge_desired=(valid_location_knowledge, ...),
    )

    available_travel_locations = Enum(
        "available_travel_locations",
        {l.upper(): l for l in agent.known_locations if l != agent.current_location},
    )
    valid_travel_actions = create_model(
        "valid_travel_actions",
        type=(Literal["travel_action"], "travel_action"),
        location=(available_travel_locations, ...),
    )

    if len(agent.known_locations) > 1:
        available_actions = Union[valid_travel_actions, valid_ask_actions]
    else:
        available_actions = valid_ask_actions

    return create_model("agent_actions", action=(available_actions, ...))


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

    available_actions = possible_next_actions(seeker_agent)
    response = seeker_agent.produce_next_action(available_actions)
    print(response)
    # available_actions = get_available_actions(seeker_agent)

    # print('available_actions:', available_actions.model_json_schema(), '\n')
    # print('seeker_agent:', seeker_agent.model_dump_json(), '\n')

    selected_action = available_actions.model_validate_json(response)
    print('selected_action:', selected_action.model_dump_json(), '\n')

    # if response = {"action": "type": "ask_action"}
    # ...send this to the knower_agent
    # .. tell the knower agent to respond format=Knowledge.model_json_schema()


if __name__ == "__main__":
    main()
