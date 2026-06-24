from openai import OpenAI
import instructor
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from openai import AzureOpenAI
from helper import get_openai_api_key, get_phoenix_endpoint, get_azure_env_var


def init_phoenix( project_name: str = "tracing-agent"):
    # Azure OpenAI client
    """
    azure_dict = get_azure_env_var()
    azure_client = AzureOpenAI(
        api_key=azure_dict["api"],
        api_version=azure_dict["version"],  # or the version you use
        azure_endpoint=azure_dict["endpoint"]
    )
    """
    openai_api = get_openai_api_key()
    client = instructor.from_openai(OpenAI())
    tool_calling_client = OpenAI(api_key=openai_api["apikey"],base_url=openai_api["baseurl"])

    # initialize the OpenAI client

    MODEL = openai_api["model"]

    PROJECT_NAME = project_name

    PHOENIX_ENDPOINT = get_phoenix_endpoint()


    tracer_provider = register(
        project_name=PROJECT_NAME,
        endpoint= PHOENIX_ENDPOINT + "v1/traces",
        
    )

    OpenAIInstrumentor().instrument(tracer_provider = tracer_provider)

    tracer = tracer_provider.get_tracer(__name__)
    
    return client, tool_calling_client, tracer