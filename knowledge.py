from openai import OpenAI
from typing import Tuple
import json
from dataclasses import dataclass

client = OpenAI()

PRIMARY_NODES_COUNT = 3
MAX_EDGES = 5
MAX_WORDS = 100
MAX_GPT_CALLS = 3


@dataclass
class KnowledgeNode:
    id: str
    fact: str


@dataclass
class KnowledgeEdge:
    nodes: Tuple[KnowledgeNode, KnowledgeNode]

    def get_other(self, node: KnowledgeNode):
        return self.nodes[0] if self.nodes[1] == node else self.nodes[1]


class KnowledgeGraph:
    primary_nodes: list[KnowledgeNode]
    nodes: list[KnowledgeNode]
    edges: list[KnowledgeEdge]

    def __init__(self, primary_nodes_count):
        self.primary_nodes = [None] * primary_nodes_count
        self.nodes = []
        self.edges = []

    def get_child_nodes(self, node: KnowledgeNode) -> list[KnowledgeNode]:
        return [e.get_other(node) for e in self.edges if node in e.nodes]

    def get_node_by_id(self, id: str) -> KnowledgeNode:
        return next(n for n in self.nodes if n.id == id)


SYSTEM_PROMPT = f"""You are having a conversation with another person. This person is either telling you facts, or asking you questions about those facts. As they tell you facts, you should store these in a database, and as they ask you questions about them, you should retrieve those facts from the database.
The database is a graph. At any given time, you can only see up to {PRIMARY_NODES_COUNT} nodes in the graph, along with all of the sibling nodes of those {PRIMARY_NODES_COUNT}. Each node can store one fact (about {MAX_WORDS} words), and can be connected to up to {MAX_EDGES} other nodes. That means the total number of fact nodes you can see at any time is up to {PRIMARY_NODES_COUNT + (PRIMARY_NODES_COUNT * MAX_EDGES)}: the {PRIMARY_NODES_COUNT} primary nodes (which may or may not be connected), and up to {MAX_EDGES} connected nodes to each of those {PRIMARY_NODES_COUNT}.
You have several tools available to you to manipulate this graph to maximize your efficiency in both storing facts and in retrieving facts to answer questions. You should use as many of these tools in any permutation you'd like to reshape the graph. Notably, if the user asks a question, you don’t have to answer it right away: you can instead respond with a series of tool calls to navigate the graph and your view of the {PRIMARY_NODES_COUNT} primary nodes. After those tool calls are executed, you will then be re-prompted with the same question and a new view of the {PRIMARY_NODES_COUNT} primary nodes in the graph.
You can call multiple tools in sequence. When you are done manipulating the knowledge graph, you should either answer the question with the answer_question tool or assert that you don't have any further response with the no_response tool.
So, for example, if the user tells you some new information, you could make a series of tool calls to add that information as multiple separate nodes in the database, including updating the primary nodes you can view via the set_node_as_primary, adding edges between related nodes with create_edge, and even synthesizing existing information across multiple nodes into new nodes.
Or, if the user asks you a question and the answer isn't in your primary nodes or their immediate siblings, you can call set_node_as_primary to navigate the graph until you you have the information available in one of your primary nodes or their sibling nodes, than you can respond with the answer_question tool. You won't get an updated view of the primary nodes until after all of your tool calls are finished executing, so you might need to issue a series of set_node_as_primary calls to navigate the graph toward a desired direction.
After any call to create_node, you should almost always then call set_node_as_primary so that you can see that node in the future. You should also liberally use the create_edge tool to create connections between related nodes.
You can currently see the following {PRIMARY_NODES_COUNT} primary nodes:"""


def get_primary_nodes_view(knowledge_graph):
    primary_nodes_view = ""
    for i, primary_node in enumerate(knowledge_graph.primary_nodes):
        primary_nodes_view += f"\n\nPrimary node slot {i}: "
        if primary_node:
            primary_nodes_view += (
                f"\nNode ID: {primary_node.id}\nNode fact: {primary_node.fact}"
            )
            for child_node in knowledge_graph.get_child_nodes(primary_node):
                primary_nodes_view += f"\nChild node ID: {child_node.id}\nChild node fact: {child_node.fact}"
        else:
            primary_nodes_view += "\nCurrently empty (no node selected)"
    return primary_nodes_view


def get_tools(knowledge_graph):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "answer_question",
                "description": "Respond with an answer to the user's question",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                    },
                    "required": ["answer"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "no_response",
                "description": "Assert that there's no question to answer and you're done manipulating the knowledge graph",
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
                "name": "create_node",
                "description": "Create a node in the knowledge graph",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "node_id": {"type": "string"},
                        "fact": {"type": "string"},
                    },
                    "required": ["node_id", "fact"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "set_node_as_primary",
                "description": "Set a node in one of the available primary slots, replacing any node currently in that slot.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "primary_slot": {
                            "type": "number",
                            "enum": list(range(len(knowledge_graph.primary_nodes))),
                        },
                        "node_id": {"type": "string"},
                    },
                    "required": ["primary_slot", "node_id"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_edge",
                "description": "Create an edge between two nodes, indicating they are related in some way.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "first_node_id": {"type": "string"},
                        "second_node_id": {"type": "string"},
                    },
                    "required": ["first_node_id", "second_node_id"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
    ]
    return tools


def tool_answer_question(knowledge_graph, args):
    return args["answer"]


def tool_create_node(knowledge_graph, args):
    print(f"Creating node '{args['node_id']}' with fact '{args['fact']}'")
    new_node = KnowledgeNode(id=args["node_id"], fact=args["fact"])
    knowledge_graph.nodes.append(new_node)


def tool_set_node_as_primary(knowledge_graph, args):
    print(f"Placing node '{args['node_id']}' in primary slot '{args['primary_slot']}'")
    knowledge_graph.primary_nodes[args["primary_slot"]] = (
        knowledge_graph.get_node_by_id(args["node_id"])
    )


def tool_create_edge(knowledge_graph, args):
    print(
        f"Creating edge between nodes '{args['first_node_id']}' and '{args['second_node_id']}'"
    )
    knowledge_graph.edges.append(
        KnowledgeEdge(
            nodes=(
                knowledge_graph.get_node_by_id(args["first_node_id"]),
                knowledge_graph.get_node_by_id(args["second_node_id"]),
            )
        )
    )


TOOL_FUNCTIONS = {
    "answer_question": tool_answer_question,
    "create_node": tool_create_node,
    "set_node_as_primary": tool_set_node_as_primary,
    "create_edge": tool_create_edge,
}


def call_gpt4o(user_input, knowledge_graph):
    print(get_primary_nodes_view(knowledge_graph))
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT + get_primary_nodes_view(knowledge_graph),
            },
            {"role": "user", "content": user_input},
        ],
        tools=get_tools(knowledge_graph),
        tool_choice="required",
    )


knowledge_graph = KnowledgeGraph(PRIMARY_NODES_COUNT)

while True:
    user_input = input()

    # Keep calling GPT-4o until it either provides an answer,
    # or affirmatively chooses not to provide an answer.
    answer = None
    will_provide_answer = True
    gpt_calls = 0
    while answer is None and will_provide_answer:
        gpt_calls += 1
        if gpt_calls > MAX_GPT_CALLS:
            print("Maximum GPT calls reached, aborting.")
            break

        completion = call_gpt4o(user_input, knowledge_graph)

        for tool_call in completion.choices[0].message.tool_calls:
            print(f"Tool call: {tool_call.function.name}")
            if tool_call.function.name == "no_response":
                will_provide_answer = False
            elif result := TOOL_FUNCTIONS[tool_call.function.name](
                knowledge_graph, json.loads(tool_call.function.arguments)
            ):
                answer = result

    if will_provide_answer:
        print(f"Response: {answer}")
    # TODO: show tools called
    # TODO: show view of graph
