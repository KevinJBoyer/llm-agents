from pydantic import BaseModel, Field, create_model, constr
from ollama import chat
from enum import Enum
from typing import Union, Literal, Optional
import json
from datetime import datetime

MODEL = "llama3.2"  # "deepseek-r1:1.5b"
MAX_CONSECUTIVE_TURNS = 3  # maximum number of turns an agent can take in a row


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

    def get_rules(self) -> str:
        return """SEEK MOTIVATION RULES (HIGHEST PRIORITY):
1. If you're in a location where you believe your sought item is:
   - You MUST IMMEDIATELY use search_action to look for it
   - Do not use any other action type when in a location with your sought item
   - Do not leave the location without searching first
2. If you don't know where your sought item is, ASK other agents about it
3. Focus your dialogue on finding your sought item
4. Only travel to locations where you think your sought item might be
5. Finding your sought item will make you happier and increase your reputation."""


class BeHelpfulMotivation(BaseModel):
    motivation_name: str = "Be helpful"

    def get_rules(self) -> str:
        return """BE HELPFUL MOTIVATION RULES (HIGHEST PRIORITY):
1. You MUST share any knowledge you have when asked
2. Be thorough when checking your knowledge to help others
3. Focus your dialogue on assisting others and sharing information, rather than asking for information yourself.
4. Doing a good job will make you more helpful and increase your reputation."""


class BaseAction(BaseModel):
    type: str
    message: str
    agent_to_talk_to: Enum
    end_turn: bool = True

    @classmethod
    def format_spec(cls) -> str:
        return ""

    @classmethod
    def get_rules(cls) -> str:
        return ""


class DialogueAction(BaseAction):
    type: Literal["dialogue_action"] = "dialogue_action"
    is_question: bool
    knowledge: Optional[Knowledge] = None
    knowledge_desired: Optional[str] = None

    @classmethod
    def format_spec(cls) -> str:
        return """DIALOGUE ACTION
   Required fields:
   - type: "dialogue_action"
   - agent_to_talk_to: (name of agent to talk to)
   - message: (your natural dialogue)
   - is_question: true when asking, false when responding
   - knowledge: (include knowledge when sharing information, null otherwise)
   - knowledge_desired: (type of item you're asking about, null when not asking)"""

    @classmethod
    def get_rules(cls) -> str:
        return """DIALOGUE RULES:
1. When someone asks you a question, you MUST respond with a direct answer (not another question):
   - If you have the knowledge they want, share it
   - If you don't have the knowledge, try to answer the question as best as you can
2. Before responding to questions about items, carefully check your knowledge list for information about the item
3. Create natural, conversational dialogue - be friendly and clear
4. Don't repeat information that was just shared
5. Share knowledge directly without extra commentary"""


class TravelAction(BaseAction):
    type: Literal["travel_action"] = "travel_action"
    location: Enum

    @classmethod
    def format_spec(cls) -> str:
        return """TRAVEL ACTION
   Required fields:
   - type: "travel_action"
   - location: (location to travel to)
   - message: (explain where and why you're going)
   - agent_to_talk_to: (who to inform about your travel)"""

    @classmethod
    def get_rules(cls) -> str:
        return """TRAVEL RULES:
1. When using travel_action, explain where and why you're traveling
2. After traveling to a location where you believe an item is, use search_action to look for it
3. When you know where your sought item is, travel there immediately"""


class SearchAction(BaseAction):
    type: Literal["search_action"] = "search_action"
    item_type: Literal["Sword", "Stone"]

    @classmethod
    def format_spec(cls) -> str:
        return """SEARCH ACTION
   Required fields:
   - type: "search_action"
   - item_type: (type of item to search for)  # MUST use this format: "Sword" or "Stone"
   - message: (describe your search)
   - agent_to_talk_to: (who to inform about your search)
   - end_turn: true/false (defaults to true)"""

    @staticmethod
    def get_rules() -> str:
        return """SEARCH RULES:
1. You can only search for items in your current location
2. If you're in a location where you believe your sought item is:
   - You MUST IMMEDIATELY use search_action to look for it
   - Do not use any other action type when in a location with your sought item
   - Do not leave the location without searching first"""


class Agent(BaseModel):
    name: str
    known_items: list[Item] = []
    known_locations: set[str] = set()
    known_agents: list[str] = []
    current_location: str
    motivation: Union[SeekMotivation, BeHelpfulMotivation]
    knowledge: list[Knowledge]
    acquired_items: list[Item] = []

    def produce_next_action(self, available_actions, global_past_actions, rejection_reason: Optional[str] = None):
        # add explicit handling for forced travel situations at the start
        # TODO: this is a hack, still want to rely on the llm to make the right decision
        if isinstance(self.motivation, SeekMotivation):
            sought_item_type = self.motivation.item.item.type
            known_location = next((
                k.knowledge.location.lower()
                for k in self.knowledge
                if k.knowledge.of.item.type == sought_item_type
            ), None)

            if known_location and known_location != self.current_location:
                # Force travel action when we know where our item is
                for agent_name in self.known_agents:
                    try:
                        return json.dumps({
                            "action": {
                                "type": "travel_action",
                                "location": known_location.upper(),
                                "message": f"Going to {known_location} to find the {sought_item_type}.",
                                "agent_to_talk_to": agent_name.upper(),
                                "end_turn": True
                            }
                        })
                    except:
                        continue

        # look through past actions to find dialogue directed at this agent
        recent_dialogue = []
        for past_action in global_past_actions[-5:]:  # look at last 5 actions
            for agent_name, action in past_action.items():
                try:
                    if (action.action.type == "dialogue_action" and
                        action.action.agent_to_talk_to.value.lower() == self.name.lower()):
                        recent_dialogue.append({
                            "from": agent_name,
                            "message": action.action.message,
                            "is_question": action.action.is_question,
                            "knowledge_desired": action.action.knowledge_desired if action.action.is_question else None,
                            "knowledge_shared": action.action.knowledge is not None
                        })
                except:
                    continue

        '''
        if recent_dialogue:
            print(f"\n📨 Recent messages to {self.name}:")
            for msg in recent_dialogue:
                print(f"   From {msg['from']}: {msg['message']}")
        '''

        prompt = f"""Choose an action based on your current state and recent dialogue:

Current state: {self.model_dump_json()}

Recent dialogue directed at you: {recent_dialogue}

CRITICAL INSTRUCTION: If you are seeking an item and you are currently in a location where you believe that item is located, you MUST use the search_action.

{self.format_available_actions()}

{self.motivation.get_rules()}

EFFICIENCY RULES:
1. Don't state intentions - just take the action
2. When you receive useful information, act on it immediately - don't acknowledge receipt
3. Don't repeat information that was just shared
4. Share knowledge directly without extra commentary
5. When you know where your sought item is, travel there immediately
6. Be proactive - if you need information to achieve your motivation, ask questions first (but never in response to a question)

{DialogueAction.get_rules()}

{TravelAction.get_rules()}

{SearchAction.get_rules()}

TURN MANAGEMENT:
1. DEFAULT BEHAVIOR: End your turn after ONE action unless you have an urgent reason to continue
2. Valid reasons to take multiple actions:
   - You MUST search immediately after traveling to a location with your sought item
   - You MUST respond to a direct question
   - You are in the middle of a critical information exchange
3. When ending turn, explain your reasoning clearly in the message

{f"PLEASE AVOID REPEATING THIS MISTAKE: {rejection_reason}" if rejection_reason else ""}

Choose your next action and respond with a valid JSON action that matches the required fields above. Make sure to use proper JSON formatting."""

        # print(prompt)
        print(f"\n🤖 {self.name} is thinking...")

        response = chat(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=MODEL,
            format=available_actions.model_json_schema(),
        )

        return response["message"]["content"]

    def receive_action(self, action_data: "agent_actions", from_agent: str):
        """
        Update agent state based on received actions from other agents.

        Args:
            action_data: The validated Pydantic model of the action
            from_agent: Name of the agent who performed the action
        """
        try:
            action = action_data.action
            action_type = action.type

            # update known_agents if we haven't seen this agent before
            if from_agent not in self.known_agents and from_agent != self.name:
                self.known_agents.append(from_agent)
                print(f"\n🤝 {self.name} met {from_agent}")

            if action_type == "search_action" and from_agent == self.name:
                # check if the item being searched for is actually in this location
                item_type = action.item_type
                location_has_item = any(
                    k.knowledge.location.lower() == self.current_location.lower() and
                    k.knowledge.of.item.type == item_type
                    for k in self.knowledge
                )

                if location_has_item:
                    # create and add the found item
                    if item_type == "Stone":
                        new_item = Item(item=Stone())
                    elif item_type == "Sword":
                        new_item = Item(item=Sword())

                    if not any(item.item.type == item_type for item in self.acquired_items):
                        self.acquired_items.append(new_item)
                        print(f"\n🎉 {self.name} found a {item_type} in {self.current_location}!")

            elif action_type == "dialogue_action":
                # update when someone shares knowledge
                new_knowledge = action.knowledge
                if new_knowledge:
                    # normalize location case
                    new_knowledge.knowledge.location = new_knowledge.knowledge.location.lower()

                    # check if we already have this knowledge
                    knowledge_exists = any(
                        k.knowledge.location.lower() == new_knowledge.knowledge.location.lower() and
                        k.knowledge.of.item.type == new_knowledge.knowledge.of.item.type
                        for k in self.knowledge
                    )

                    if not knowledge_exists:
                        self.knowledge.append(new_knowledge)
                        # add the location to known locations (normalized)
                        self.known_locations.add(new_knowledge.knowledge.location)
                        print(f"\n🗺️  {self.name} discovered new location: {new_knowledge.knowledge.location}")
                        # add the item type to known items if we don't have it
                        new_item = new_knowledge.knowledge.of
                        if not any(item.item.type == new_item.item.type for item in self.known_items):
                            self.known_items.append(new_item)
                        print(f"\n💡 {self.name} learned: {new_knowledge.knowledge.of.item.type} is in {new_knowledge.knowledge.location}")

                # track items mentioned in questions
                if action.is_question and action.knowledge_desired:
                    knowledge_desired = action.knowledge_desired
                    if not any(item.item.type == knowledge_desired for item in self.known_items):
                        if knowledge_desired == "Stone":
                            self.known_items.append(Item(item=Stone()))
                        elif knowledge_desired == "Sword":
                            self.known_items.append(Item(item=Sword()))

            elif action_type == "travel_action":
                # learn about new locations from observing travel
                new_location = action.location.value.lower()
                if new_location not in self.known_locations:
                    self.known_locations.add(new_location)
                    print(f"\n🗺️  {self.name} discovered new location: {new_location}")

                # update current_location if this is our own travel action
                if from_agent == self.name:
                    self.current_location = new_location
                    print(f"\n👣 {self.name} moved to {new_location}")

        except Exception as e:
            print(f"\n❌ Error processing action for {self.name}: {e}")

    def print_state(self):
        """Pretty print the current state of the agent"""
        print(f"\n📊 {self.name}'s Current State:")
        print(f"   📍 Location: {self.current_location}")

        if self.acquired_items:
            print(f"   🎒 Inventory: {', '.join(item.item.type for item in self.acquired_items)}")

        if isinstance(self.motivation, SeekMotivation):
            print(f"   🎯 Seeking: {self.motivation.item.item.type}")
        else:
            print(f"   🤝 Motivation: {self.motivation.motivation_name}")

        if self.knowledge:
            print("   💡 Knowledge:")
            for k in self.knowledge:
                print(f"      • {k.knowledge.of.item.type} is in {k.knowledge.location}")

        if self.known_locations:
            print(f"   🗺️  Known locations: {', '.join(sorted(self.known_locations))}")

        if self.known_agents:
            print(f"   👥 Known agents: {', '.join(sorted(self.known_agents))}")

    @staticmethod
    def format_available_actions() -> str:
        actions = [DialogueAction, TravelAction, SearchAction]
        return "Available Actions:\n\n" + "\n\n".join(
            f"{i+1}. {action.format_spec()}"
            for i, action in enumerate(actions)
        )


def possible_next_actions(agent: Agent):
    # TODO: pylance is very mad about this
    available_locations = Enum(
        "available_locations",
        {l.upper(): l for l in agent.known_locations}
    )

    available_agents_to_talk_to = Enum(
        "available_agents_to_talk_to",
        {l.upper(): l for l in agent.known_agents if l.lower() != agent.name.lower()}
    )

    class ValidDialogueAction(DialogueAction):
        agent_to_talk_to: available_agents_to_talk_to
        message: constr(min_length=1, strip_whitespace=True)

    class ValidTravelAction(TravelAction):
        location: available_locations
        message: constr(min_length=1, strip_whitespace=True)
        agent_to_talk_to: available_agents_to_talk_to

    class ValidSearchAction(SearchAction):
        message: constr(min_length=1, strip_whitespace=True)
        agent_to_talk_to: available_agents_to_talk_to

    if len(agent.known_locations) > 1:
        available_actions = Union[ValidTravelAction, ValidDialogueAction, ValidSearchAction]
    else:
        available_actions = Union[ValidDialogueAction, ValidSearchAction]

    return create_model("agent_actions", action=(available_actions, ...))


class ValidationResult(BaseModel):
    is_valid: bool = Field(description="Whether the action is valid in the current world state")
    reason: str = Field(description="Detailed explanation of why the action is valid or invalid")


class Supervisor:
    def validate_action(self, agent: Agent, action_data: "agent_actions", global_past_actions: list) -> tuple[bool, str]:
        """
        Uses LLM to validate if an action is logically consistent with the current world state.

        Returns:
            tuple[bool, str]: (is_valid, reason)
        """
        action = action_data.action

        # Add specific validation for travel actions
        if action.type == "travel_action":
            destination = action.location.value.lower()
            if destination not in agent.known_locations:
                return False, f"Invalid travel destination: '{action.location.value}' is not a known location"

        # Add validation for seeking agents who know their item's location
        if isinstance(agent.motivation, SeekMotivation):
            sought_item_type = agent.motivation.item.item.type
            known_location = next((
                k.knowledge.location.lower()
                for k in agent.knowledge
                if k.knowledge.of.item.type == sought_item_type
            ), None)

            if known_location and known_location != agent.current_location:
                if action.type != "travel_action" or action.location.value.lower() != known_location:
                    return False, f"Agent must use travel_action to move to {known_location} where they know their sought item ({sought_item_type}) is located before taking other actions"

        # Add validation for stated intentions without actions
        if action.type == "dialogue_action":
            message_lower = action.message.lower()
            intention_phrases = [
                "going to",
                "will check",
                "will go",
                "plan to",
                "intend to",
                "about to",
                "planning to"
            ]
            if any(phrase in message_lower for phrase in intention_phrases):
                return False, "Don't state intentions - just take the action. Instead of saying what you will do, do it."

        prompt = f"""As a world supervisor, evaluate if this action is logically consistent and makes sense for the agent to perform, given the current world state.

CURRENT WORLD STATE:
Agent: {agent.model_dump_json()}
Action: {action_data.model_dump_json()}
Recent history (last 5 actions): {global_past_actions[-5:]}

VALIDATION RULES:
1. Travel actions must only target known locations (not agent names)
2. Search actions can only be performed in the agent's current location
3. Dialogue actions must target known agents
4. All actions must be logically consistent with the agent's knowledge and goals
5. For seeking agents: asking about items, traveling to find them, and searching locations are valid strategies
6. For helpful agents: sharing knowledge and responding to questions are valid behaviors
7. CRITICAL: If a seeking agent knows where their sought item is, they MUST travel there immediately before taking any other actions
8. CRITICAL: Agents must not state intentions without taking action - instead of saying what they will do, they should do it

Consider:
- Is this action logically possible given the agent's current state and knowledge?
- Does it help the agent progress toward their goals?
- Is the action consistent with the agent's recent interactions?
- Would this action make sense to an outside observer?
- Does the action actually do something instead of just stating intentions?

IMPORTANT: end_turn actions should almost always be considered valid, as agents are allowed to end their turn at any time. Only reject end_turn actions if they contain clearly inappropriate messages.

Analyze the action and respond with a JSON object indicating if the action is valid and why:
{{
    "is_valid": true/false,
    "reason": "Detailed explanation of why the action is valid or invalid"
}}"""

        response = chat(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=MODEL,
            format=ValidationResult.model_json_schema()
        )

        try:
            result = ValidationResult.model_validate_json(response["message"]["content"])
            return result.is_valid, result.reason
        except Exception as e:
            return False, f"Error parsing supervisor response: {str(e)}"


def main():
    seeker_agent = Agent(
        name="Seeker",
        current_location="city",
        known_items=[Item(item=Stone()), Item(item=Sword())],
        known_locations={"city"},
        known_agents=["Knower"],
        motivation=SeekMotivation(item=Item(item=Sword())),
        knowledge=[],
    )
    knower_agent = Agent(
        name="Knower",
        current_location="city",
        known_locations={"city", "field"},
        known_agents=["Seeker"],
        motivation=BeHelpfulMotivation(),
        knowledge=[
            Knowledge(knowledge=Location(of=Item(item=Sword()), location="field"))
        ],
    )

    agents = [seeker_agent, knower_agent]
    supervisor = Supervisor()
    global_past_actions = []

    while True:
        for agent in agents:
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"\n\n{'='*50}")
            print(f"🎯 {agent.name}'s turn @ {current_time}")
            print(f"{'='*50}")

            turn_count = 0
            while turn_count < MAX_CONSECUTIVE_TURNS:
                available_actions = possible_next_actions(agent)

                rejection_reason = None
                while True:  # keep trying until we get a valid action
                    response = agent.produce_next_action(available_actions, global_past_actions, rejection_reason)

                    try:
                        validated_action = available_actions.model_validate_json(response)
                        action = validated_action.action

                        # check if action is logically valid
                        is_valid, reason = supervisor.validate_action(agent, validated_action, global_past_actions)
                        if not is_valid:
                            print(f"\n⚖️ Supervisor rejected action: {reason}")
                            print(f"❌ Rejected action: {validated_action.model_dump_json(indent=2)}")
                            rejection_reason = reason
                            continue  # Try again

                        global_past_actions.append({agent.name: validated_action})

                        agent.print_state()

                        print(f"\n🔄 {agent.name}'s action:")
                        if action.type == "dialogue_action":
                            print(f"   💬 To {action.agent_to_talk_to.value}: {action.message}")
                            if action.knowledge:
                                print(f"   📢 Sharing knowledge: {action.knowledge.knowledge.of.item.type} is in {action.knowledge.knowledge.location}")
                        elif action.type == "travel_action":
                            print(f"   🚶 Moving to: {action.location.value}")
                            print(f"   💬 To {action.agent_to_talk_to.value}: {action.message}")
                        elif action.type == "search_action":
                            print(f"   🔍 Searching for: {action.item_type}")
                            print(f"   💬 To {action.agent_to_talk_to.value}: {action.message}")

                        for other_agent in agents:
                            other_agent.receive_action(validated_action, agent.name)

                        if action.end_turn:
                            break  # exit the action validation loop

                        break  # valid action processed, continue to next turn

                    except Exception as e:
                        print(f"\n❌ Error: {str(e)}")
                        print("\n🔍 Full response that caused error:")
                        print(f"{response}")
                        continue

                if action.end_turn:
                    break  # exit the turn loop

                turn_count += 1
                if turn_count < MAX_CONSECUTIVE_TURNS:
                    print(f"\n📝 {agent.name} has {MAX_CONSECUTIVE_TURNS - turn_count} turns remaining")
                else:
                    print(f"\n⏰ {agent.name} has reached maximum consecutive turns")

            # input('\npress ENTER to continue...')


if __name__ == "__main__":
    main()
