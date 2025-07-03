import asyncio
import logging
import json
from typing import AsyncGenerator, List, Dict, Any

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from typing_extensions import override

# --- Configure Logging for Clear Output ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
APP_NAME = "orchestrator_app"
USER_ID = "test_user"
SESSION_ID = "session_1"
# Replace with your preferred Gemini model
GEMINI_MODEL = "gemini-1.5-flash"

# --- Sub-Agents Definition ---

def create_planner_agent() -> LlmAgent:
    """
    Creates the PlannerAgent.
    This agent is responsible for taking a high-level task and breaking it down
    into a series of concrete, executable steps. The output is a JSON list of strings.
    """
    return LlmAgent(
        name="PlannerAgent",
        model=GEMINI_MODEL,
        instruction="""
You are a master planner. Your role is to take a user's request and break it down into a
clear, step-by-step plan. Each step should be a distinct action.
The user's request will be in the session state under the key 'user_request'.
You MUST output the plan as a JSON object with a single key "plan" which contains a list of strings.
For example: {"plan": ["step 1 description", "step 2 description", "step 3 description"]}
Do not add any other text in your response.
""",
        output_key="plan_output",  # The key to store the raw LLM output in the session state
    )

def create_execution_agent() -> LlmAgent:
    """
    Creates the ExecutionAgent.
    This agent takes a single step from the plan and executes it, producing a result.
    """
    return LlmAgent(
        name="ExecutionAgent",
        model=GEMINI_MODEL,
        instruction="""
You are an expert executor. You will be given a single task to perform.
The task is provided in the session state under the key 'current_step'.
Your job is to execute this task and provide a concise result.
The history of what has been done is in the 'execution_history' state key.
Output only the result of the current step's execution.
""",
        output_key="step_result", # The key to store the result of the step in the session state
    )

# --- Custom Orchestrator Agent ---

class OrchestratorAgent(BaseAgent):
    """
    Orchestrates the workflow between a PlannerAgent and an ExecutionAgent.

    This agent manages the overall process of receiving a user request,
    generating a plan, executing each step of the plan, and producing a
    final, synthesized result. It demonstrates custom, stateful orchestration. [1]
    """
    # Pydantic model fields for type checking and initialization
    planner: LlmAgent
    executor: LlmAgent

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, name: str, planner: LlmAgent, executor: LlmAgent):
        """
        Initializes the OrchestratorAgent.

        Args:
            name: The name of the agent.
            planner: An instance of the PlannerAgent.
            executor: An instance of the ExecutionAgent.
        """
        # The sub_agents list informs the ADK framework of the agent hierarchy. [1]
        super().__init__(
            name=name,
            planner=planner,
            executor=executor,
            sub_agents=[planner, executor],
        )

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Implements the custom orchestration logic for the agent. [1]
        """
        logger.info(f"[{self.name}] Starting orchestration workflow.")

        # 1. --- Run PlannerAgent to create the plan ---
        logger.info(f"[{self.name}] Invoking PlannerAgent to create a plan.")
        async for event in self.planner.run_async(ctx):
            yield event

        # Extract and validate the plan from session state
        plan_output_str = ctx.session.state.get("plan_output")
        if not plan_output_str:
            logger.error(f"[{self.name}] PlannerAgent failed to produce output. Aborting.")
            return

        try:
            plan_data = json.loads(plan_output_str)
            plan: List[str] = plan_data["plan"]
            if not isinstance(plan, list) or not all(isinstance(step, str) for step in plan):
                raise ValueError("Plan is not a list of strings.")
            ctx.session.state["plan"] = plan
            logger.info(f"[{self.name}] Plan created successfully: {plan}")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"[{self.name}] Failed to parse plan from PlannerAgent: {e}. Output was: {plan_output_str}")
            return

        # 2. --- Execute the plan step-by-step ---
        execution_history: List[Dict[str, Any]] = []
        ctx.session.state["execution_history"] = execution_history

        for i, step in enumerate(plan):
            logger.info(f"[{self.name}] Executing step {i+1}/{len(plan)}: {step}")
            ctx.session.state["current_step"] = step

            # Invoke the ExecutionAgent
            async for event in self.executor.run_async(ctx):
                yield event

            step_result = ctx.session.state.get("step_result")
            if not step_result:
                logger.warning(f"[{self.name}] ExecutionAgent did not produce a result for step: {step}")
                step_result = "No result produced."

            # Update execution history
            execution_history.append({"step": step, "result": step_result})
            logger.info(f"[{self.name}] Step {i+1} result: {step_result}")

        logger.info(f"[{self.name}] All steps executed. Finalizing.")

        # 3. --- Synthesize the final answer ---
        # In a real-world scenario, you might have another LLM call to synthesize
        # the results from the execution_history. For this example, we'll format it.
        final_answer = self._synthesize_final_answer(ctx)
        logger.info(f"[{self.name}] Final synthesized answer:\n{final_answer}")

        # Yield a final event with the result
        final_event = Event.create_final_response(
            author=self.name,
            content=types.Content(parts=[types.Part(text=final_answer)])
        )
        yield final_event

    def _synthesize_final_answer(self, ctx: InvocationContext) -> str:
        """Creates a final summary from the execution history."""
        history = ctx.session.state.get("execution_history", [])
        if not history:
            return "The process completed, but no results were generated."

        summary = "The following plan was executed:\n\n"
        for item in history:
            summary += f"- **Step:** {item['step']}\n"
            summary += f"  - **Result:** {item['result']}\n\n"
        return summary


# --- Main Execution Logic ---

async def main():
    """
    Sets up the agents, runner, and session, then runs the orchestration.
    """
    # 1. Instantiate the agents
    planner = create_planner_agent()
    executor = create_execution_agent()
    orchestrator = OrchestratorAgent(
        name="OrchestratorAgent",
        planner=planner,
        executor=executor
    )

    # 2. Setup Runner and Session Service [3]
    session_service = InMemorySessionService()
    runner = Runner(
        agent=orchestrator,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # 3. Define the initial user request and session state
    user_request = "Write a short blog post about the benefits of learning a new programming language in 2025."
    initial_state = {"user_request": user_request}

    # Create a session with the initial state
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state=initial_state
    )

    logger.info(f"Starting agent with request: '{user_request}'")

    # 4. Run the agent and process events
    final_response = "No final response captured."
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        # The content of new_message can be used to pass initial input,
        # but here we pre-populate the state for clarity.
        new_message=types.Content(parts=[types.Part(text=user_request)])
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_response = event.content.parts[0].text
            logger.info(f"--- Final Response from {event.author} ---")
            print(final_response)

    # 5. Inspect the final session state
    final_session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    logger.info("\n--- Final Session State ---")
    print(json.dumps(final_session.state, indent=2))


if __name__ == "__main__":
    # In a standalone script, you run the async main function like this.
    # In a Jupyter/Colab notebook, you can just `await main()`.
    asyncio.run(main())
