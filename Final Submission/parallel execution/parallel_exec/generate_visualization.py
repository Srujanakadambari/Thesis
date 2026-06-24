from prompts.default import (
    CHART_CONFIGURATION_PROMPT,
    CREATE_CHART_PROMPT,
    CREATE_CHART_TYPE_JUSTIFICATION_PROMPT,
    CORRECTION_PROMPT,
    CONFIG_CORRECTION_PROMPT,  # kept (even if unused)
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Union

from opentelemetry.trace import StatusCode
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import ValidationError


# ----------------------------
# Safe tracer wrapper (works even when tracer is None)
# ----------------------------
class NoOpTracer:
    def tool(self, fn):
        return fn

    def chain(self, fn):
        return fn

    def start_as_current_span(self, *args, **kwargs):
        class _CM:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                return False

            def set_input(self_inner, value=None, **kw):
                pass

            def set_output(self_inner, value=None, **kw):
                pass

            def set_status(self_inner, value=None, **kw):
                pass

        return _CM()


# ----------------------------
# Helpers
# ----------------------------
def _to_dict(x: Any) -> Any:
    """Convert Pydantic models to dict safely. Leave dicts/strings as-is."""
    if hasattr(x, "model_dump"):
        return x.model_dump()
    if hasattr(x, "dict"):
        return x.dict()
    return x


def _msg_to_dict(msg: Any) -> Dict[str, Any]:
    """Convert OpenAI message objects to plain dict so prompts stay stable."""
    if isinstance(msg, dict):
        return msg

    if hasattr(msg, "model_dump"):
        d = msg.model_dump()
        out = {"role": d.get("role"), "content": d.get("content")}
        if d.get("tool_calls") is not None:
            out["tool_calls"] = d.get("tool_calls")
        return out

    # fallback attribute access
    role = getattr(msg, "role", None)
    content = getattr(msg, "content", None)
    tool_calls = getattr(msg, "tool_calls", None)
    out = {"role": role, "content": content}
    if tool_calls is not None:
        out["tool_calls"] = tool_calls
    return out


def call_instructor(client, model: str, prompt: str, response_model):
    """
    Version-safe instructor call:
    - never passes unsupported kwargs like return_response
    - supports instructor versions returning either:
      * parsed_model
      * (parsed_model, raw_response)
    """
    result = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_model=response_model,
    )

    # some instructor versions return (parsed, raw)
    if isinstance(result, tuple) and len(result) == 2:
        parsed, _raw = result
        return parsed

    return result


# ----------------------------
# Agent
# ----------------------------
class Agent:
    def __init__(self, client, tool_calling_client, tracer, model):
        self.client = client
        self.tool_calling_client = tool_calling_client
        self.tracer = tracer if tracer is not None else NoOpTracer()
        self.model = model

        # instrument safely
        self.generate_visualization = self.tracer.tool(self.generate_visualization)
        self.analyze_data = self.tracer.chain(self.analyze_data)
        self.extract_chart_config = self.tracer.chain(self.extract_chart_config)
        self.justify_chart_type = self.tracer.chain(self.justify_chart_type)
        self.generate_final_response = self.tracer.chain(self.generate_final_response)

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
                            "data": {"type": "string", "description": "The data as a markdown table"},
                            "user_query": {"type": "string", "description": "The users query"},
                        },
                        "required": ["data", "user_query"],
                    },
                },
            },
        ]

        self.tool_implementations = {
            "generate_visualization": self.generate_visualization,
        }

    # ----------------------------
    # Tool implementation
    # ----------------------------
    def generate_visualization(self, data: str, user_query: str) -> Dict[str, Any]:
        # Step 1: Data Analysis (sequential — both downstream steps depend on this)
        analysis = self.analyze_data(data, user_query)

        # Steps 2 & 3: Chart Config Extraction + Chart Type Justification
        # These are independent of each other — run in parallel.
        # Critical path is extract_chart_config (~19s) > justify_chart_type (~7s),
        # so total wait = max(19s, 7s) ≈ 19s instead of 19s + 7s = 26s.
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_config = executor.submit(self.extract_chart_config, data, analysis)
            future_type = executor.submit(self.justify_chart_type, analysis, data)

            config = future_config.result()
            chart_type = future_type.result()

        config["charttype"] = chart_type
        return config

    # ----------------------------
    # Instructor structured calls
    # ----------------------------
    def analyze_data(self, data: str, query: str) -> Dict[str, Any]:
        formatted_prompt = DATA_ANALYSIS_PROMPT.format(data=data, query=query)
        parsed = call_instructor(self.client, self.model, formatted_prompt, DataAnalysis)
        return _to_dict(parsed)

    def extract_chart_config(self, data: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        formatted_prompt = CHART_CONFIGURATION_PROMPT.format(data=data, analysis=analysis)
        parsed = call_instructor(self.client, self.model, formatted_prompt, VisualizationConfig)

        res = _to_dict(parsed)

        # charttype could be Enum in some outputs
        ct = res.get("charttype")
        if isinstance(ct, ChartType):
            res["charttype"] = ct.value
        elif hasattr(ct, "value"):
            res["charttype"] = ct.value

        return res

    @retry(
        retry=retry_if_exception_type(ValidationError),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def justify_chart_type(self, analysis: Dict[str, Any], data: str) -> str:
        charttypes = {ct.name for ct in ChartType}
        formatted_prompt = CREATE_CHART_TYPE_JUSTIFICATION_PROMPT.format(
            charttypes=charttypes,
            analysis=analysis,
            data=data,
        )

        parsed = call_instructor(self.client, self.model, formatted_prompt, ChartTypeJustification)

        # typical pydantic: parsed.chart_type is Enum
        if hasattr(parsed, "chart_type") and parsed.chart_type is not None:
            return parsed.chart_type.value

        d = _to_dict(parsed)
        ct = d.get("chart_type")
        if isinstance(ct, ChartType):
            return ct.value
        return ct

    # Optional steps (kept runnable)
    def correct_config(self, config: dict) -> dict:
        formatted_prompt = CORRECTION_PROMPT.format(config=config)
        parsed = call_instructor(self.client, self.model, formatted_prompt, CorrectionList)
        return _to_dict(parsed)

    def create_chart(self, charttype: str, config: dict, correctedconfig: dict) -> str:
        formatted_prompt = CREATE_CHART_PROMPT.format(
            charttype=charttype, config=config, correctedconfig=correctedconfig
        )
        parsed = call_instructor(self.client, self.model, formatted_prompt, VisualizationCode)

        # response model may store code in .code
        code = getattr(parsed, "code", None)
        if code is None:
            d = _to_dict(parsed)
            code = d.get("code", "")

        code = code.replace("```python", "").replace("```", "").strip()
        return code

    # ----------------------------
    # Tool-call execution
    # ----------------------------
    def handle_tool_calls(self, tool_calls, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            function = self.tool_implementations[tool_name]
            function_args = json.loads(tool_call.function.arguments)

            result = function(**function_args)  # dict result

            # tool message content MUST be string
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

        return messages

    def generate_final_response(self, messages: List[Dict[str, Any]]) -> str:
        formatted_prompt = FINAL_RESPONSE_PROMPT.format(messages=messages)
        parsed = call_instructor(self.client, self.model, formatted_prompt, FinalResponse)

        # Some models store it as parsed.final_response
        if hasattr(parsed, "final_response"):
            return parsed.final_response

        d = _to_dict(parsed)
        return d.get("final_response", json.dumps(d))

    # ----------------------------
    # Router loop
    # ----------------------------
    def run_agent(self, messages: Union[List[Dict[str, Any]], str]):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # Ensure system prompt exists AND is first
        if not any(isinstance(m, dict) and m.get("role") == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        with self.tracer.start_as_current_span("router_call", openinference_span_kind="chain") as span:
            span.set_input(value=messages)

            response = self.tool_calling_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
            )

            msg = response.choices[0].message
            messages.append(_msg_to_dict(msg))

            tool_calls = getattr(msg, "tool_calls", None)
            span.set_status(StatusCode.OK)

            if tool_calls:
                messages = self.handle_tool_calls(tool_calls, messages)
                span.set_output(value={"tool_called": True})
                # Return the last tool result (string) or you can return dict by parsing it back
                return messages[-1]["content"]

            # No tool calls -> final response
            final = self.generate_final_response(messages)
            span.set_output(value=final)
            return final

    def start_main_span(self, messages: Union[List[Dict[str, Any]], str]):
        with self.tracer.start_as_current_span("AgentRun", openinference_span_kind="agent") as span:
            span.set_input(value=messages)
            ret = self.run_agent(messages)
            span.set_output(value=ret)
            span.set_status(StatusCode.OK)
            return ret
