
SYSTEM_PROMPT = """
    You are a helpful assistant that is an expert in data analysis and visualization.
"""

DATA_ANALYSIS_PROMPT = """
Analyze the following data: {data}
Your job is to answer the following question: {query}
Describe all interesting insights and observations you find in the data.
Also note the users extra-wishes in short terms, that can be accomplished in a chart.
"""

CHART_CONFIGURATION_PROMPT = """
Generate a chart configuration based on this data: {data}
The chat should be annotated according to this analysis: {analysis}
Please consider, that the output should be compitable with matplotlib.
The ChartDataGroups need to be treated as one label group.
Please consider using the data_id and data_value for annotating, when you want to annotate
a specific item. data_id is the index of the data list and the data_value the corresponding
value of the data from data_id. Pie, Stackedbar and Columncharts need it. 
If the chart is a Pie the x_data is ignored. One datagroup is used for one pie piece.
"""
#If you want to make a Stackedbarchart consider the height, after all y_values of each bar with the same
#x_values are added on each other

### Added Idea from https://arxiv.org/pdf/2507.14819

CORRECTION_PROMPT = """
Based on this config: {config} find possible problems regarding data errors.
Correct them divided into the config fields and also give a score from 0-10 how negible it is. While 0 being negibale and 10 is problematic.
"""

CONFIG_CORRECTION_PROMPT = """
Based on this config: {config} and the correction: {correction} correct the config.
Correct all problems.
"""
#PROBLEM USER WISHES ARENT SPECIFIDED IN JUSTIFICATION
CREATE_CHART_TYPE_JUSTIFICATION_PROMPT = """
Based on this data: {data} and the analysis: {analysis} generate a short justification for the best chart-type of: {charttypes}
The useres wishes in the analysis should have a higher priority in chosing the best chart type.
"""

####
#Try to include all: {userwishes} that are possible.
CREATE_CHART_PROMPT = """
Write python code to create a chart based on the following configuration: {config} with the following correction: {correctedconfig}
Create a chart with the type: {charttype}
The annotations and legend/key of the config should not intersect with each other like e.g. (text, arrows, chart-lines,..., chart title)
Only return the code, no other text.
"""
FINAL_RESPONSE_PROMPT = """
Generate a final, structured response for the user.
The context is: {messages}
"""