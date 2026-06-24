from prompts.default import (
    CHART_CONFIGURATION_PROMPT,
    CREATE_CHART_PROMPT,
    CREATE_CHART_TYPE_JUSTIFICATION_PROMPT,
    CORRECTION_PROMPT,
    CONFIG_CORRECTION_PROMPT,
    DATA_ANALYSIS_PROMPT,
    FINAL_RESPONSE_PROMPT,
    SYSTEM_PROMPT,
)
from response_models.default import (
    VisualizationConfig,
    VisualizationCode,
    DataAnalysis,
    FinalResponse,
    ChartTypeJustification,
    ChartType,
    CorrectionList,
)
import json
from opentelemetry.trace import StatusCode
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import ValidationError


class Agent:
    def __init__(self, client, tool_calling_client, tracer, model):
        self.client = client
        self.tool_calling_client = tool_calling_client
        self.tracer = tracer
        self.model = model

        self.generate_visualization = self.tracer.tool(self.generate_visualization)
        self.analyze_data = self.tracer.chain(self.analyze_data)
        self.extract_chart_config = self.tracer.chain(self.extract_chart_config)
        #self.correct_config = self.tracer.chain(self.correct_config)
        self.justify_chart_type = self.tracer.chain(self.justify_chart_type)
        #self.create_chart = self.tracer.chain(self.create_chart)

        self.default_llm_config = {"model": self.model}

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "generate_visualization",
                    "description": "Analyze the data and generate a visualization",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "string",
                                "description": "The data as a markdown table",
                            },
                            "user_query": {
                                "type": "string",
                                "description": "The users query",
                            },
                        },
                        "required": ["data", "user_query"],
                    },
                },
            },
        ]
        # Dictionary mapping function names to their implementations
        self.tool_implementations = {
            "generate_visualization": self.generate_visualization,
        }

    def generate_visualization(self, data: str, user_query: str) -> str:
        """Generate a visualization based on the data and goalS    
        """
        analysis = self.analyze_data(data, user_query)
        config = self.extract_chart_config(data, analysis=analysis)
        #corrected_config = self.correct_config(config=config)
        config["charttype"]  = self.justify_chart_type(analysis=analysis,data=data)
        #code = self.create_chart(charttype,corrected_config,config) # maybe add userwishes currently in 
        return config

    def analyze_data(self, data: str, query: str) -> dict:
        """Analyze the data"""
        formatted_prompt = DATA_ANALYSIS_PROMPT.format(data=data, query=query)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": formatted_prompt}],
            response_model=DataAnalysis,
        )

        return response.analysis

    def extract_chart_config(self, data: str, analysis: str) -> dict:
        """Generate chart visualization configuration

        Args:
            data: String containing the data to visualize
            analysis: Description of what the visualization consider for the annotations

        Returns:
            Dictionary containing chart configuration
        """
        formatted_prompt = CHART_CONFIGURATION_PROMPT.format(
            data=data, analysis=analysis
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": formatted_prompt}],
            response_model=VisualizationConfig,
        )
        res = response.model_dump()
        res["charttype"] = res["charttype"].value
        return res

    @retry(
        retry=retry_if_exception_type(ValidationError),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )

    def justify_chart_type(self,analysis: dict, data: str) -> str:
        """Create a Justification for Chart Types with a score"""
        charttypes = {chat_type.name for chat_type in ChartType}
        formatted_prompt = CREATE_CHART_TYPE_JUSTIFICATION_PROMPT.format(charttypes=charttypes,analysis=analysis, data = data)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": formatted_prompt}],
            response_model=ChartTypeJustification,
        )
        return response.chart_type.value

    def correct_config(self, config: dict) -> dict:
        """Correct the chart-config if necessary"""
        formatted_prompt = CORRECTION_PROMPT.format(config=config)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": formatted_prompt}],
            response_model=CorrectionList,
        )
        return response.model_dump()


    def create_chart(self, charttype: str, config: dict, correctedconfig:dict) -> str:
        """Create a chart based on the configuration"""
        formatted_prompt = CREATE_CHART_PROMPT.format(charttype=charttype, config=config, correctedconfig=correctedconfig)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": formatted_prompt}],
            response_model=VisualizationCode,
        )

        code = response.code
        code = code.replace("```python", "").replace("```", "")
        code = code.strip()

        return code

    # code for executing the tools returned in the model's response
    def handle_tool_calls(self, tool_calls, messages) -> tuple[list, str]:

        for tool_call in tool_calls:
            function = self.tool_implementations[tool_call.function.name]
            function_args = json.loads(tool_call.function.arguments)
            result = function(**function_args)
            messages.append(
                {"role": "tool", "content": result, "tool_call_id": tool_call.id}
            )
            print(tool_call.function.name)

        return messages

    def generate_final_response(self, messages: list) -> str:
        """Create a final response to the user's question"""
        formatted_prompt = FINAL_RESPONSE_PROMPT.format(messages=messages)

        response = self.client.chat.completions.create(
            **self.default_llm_config,
            messages=[{"role": "user", "content": formatted_prompt}],
            response_model=FinalResponse,
        )

        return response

    def run_agent(self, messages):
        print("Running agent with messages:", messages)

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # Check and add system prompt if needed
        if not any(
            isinstance(message, dict) and message.get("role") == "system"
            for message in messages
        ):
            system_prompt = {"role": "system", "content": SYSTEM_PROMPT}
            messages.append(system_prompt)

        while True:
            # Router Span
            print("Starting router call span")
            with self.tracer.start_as_current_span(
                "router_call",
                openinference_span_kind="chain",
            ) as span:
                span.set_input(value=messages)
                print("Making router call to OpenAI")
                response = self.tool_calling_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                )
                messages.append(response.choices[0].message)
                tool_calls = response.choices[0].message.tool_calls
                print("Received response with tool calls:", bool(tool_calls))
                span.set_status(StatusCode.OK)

                # if the model decides to call function(s), call handle_tool_calls
                if tool_calls:
                    print("Starting tool calls span")
                    messages = self.handle_tool_calls(tool_calls, messages)
                    span.set_output(value=tool_calls)
                    return messages[-1]
                else:
                    print("No tool calls, returning final response")
                    response = self.generate_final_response(messages)
                    span.set_output(value=response)
                    return response

    def start_main_span(self, messages):
        print("Starting main span with messages:", messages)

        with self.tracer.start_as_current_span(
            "AgentRun", openinference_span_kind="agent"
        ) as span:
            span.set_input(value=messages)
            ret = self.run_agent(messages)
            print("Main span completed with return value:", ret)
            span.set_output(value=ret)
            span.set_status(StatusCode.OK)
            return ret
