from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import get_openai_callback
from langchain.prompts import MessagesPlaceholder
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.tools import tool
from dotenv import load_dotenv
import utils

load_dotenv()


@tool
def generate_response(question: str) -> str:
    """
    Useful to generate a response. Applies if the user asks questions about their life, work or relationships. 
    """
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    chain = LLMChain(
        llm=llm,
    )
    return chain.run(input=question)


class heal_agent():

    def __init__(self) -> None:
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        memory = ConversationSummaryBufferMemory(
            memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

        system_message = SystemMessage(
            content="""You are a world class therapist to help people heal from broken work, life and relationships. 
                """
        )

        agent_kwargs = {
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
            "system_message": system_message,
        }

        # tools = [summarize_youtube_video, summarize_article]
        tools = [generate_response]

        self.agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            agent_kwargs=agent_kwargs,
            memory=memory,
        )

    def generate(self, query):
        with get_openai_callback() as cb:
            output = self.agent.run(input=query)
            print(output)
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost}")
            utils.log_openai_usage(
                cb.total_tokens, cb.prompt_tokens, cb.completion_tokens, cb.total_cost)
            return output


if __name__ == "__main__":
    print("input something...")
    agent = heal_agent()
    query = input()
    print(agent.generate(query))
