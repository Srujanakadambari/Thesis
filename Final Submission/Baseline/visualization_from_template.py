import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.axes._axes import Axes
from matplotlib.patches import Wedge
from matplotlib.text import Text
from matplotlib.patches import Rectangle
from matplotlib.container import BarContainer
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
import numpy as np
from numpy.ma import MaskedArray
import random
import json
import copy
from pydantic import BaseModel, Field, field_validator, ValidationError

#Colors that represents NV style
COLOR_STACK = ["#30599B","#5E86C5","#3C547B","#263753"]
TITLE_LENGTH_BEFORE_LB = 35
ANNO_LENGTH_BEFORE_LB = 50
PIE_ANNO_OFFSET = 0.7
colorstack: list[str]
axes: Axes
config: dict



#------Wrappers for charts-------#

def handle_chart_type():
  global config
  chart_type = config["charttype"]

  items_for_style = []

  if chart_type != None:
      if chart_type == "bar" or chart_type == "column" or chart_type == "stackedbar":
          items = configure_bar()
          handle_config(items=items,needs_styling=True)

      elif chart_type == "line":
          items = configure_plot()
          handle_config(items=items,needs_styling=True)
          
      elif chart_type == "scatter":
          items = configure_scatter()
          handle_config(items=items)
          
      elif chart_type == "pie":
          items: list = configure_pie()
          handle_config(needs_annotations=True,needs_axes=False,items=items)

def configure_bar()->list:
  global axes, config
  bars = []
  data: dict = config["data"]
  NORMAL_WIDTH = 0.8
  columnchart = True

  if config["charttype"] == "stackedbar":
    columnchart = False

  #Get bars for each x_value
  bar_groups_count = {}
  for dat in data:
    for d in dat["x_data"]:
      key = str(d)
      if key in bar_groups_count:
        bar_groups_count[key] += 1
      else:
        bar_groups_count[key] = 1

  #Get max value
  maxbar = 1
  maxbar = max(bar_groups_count.values())
  xshift = -NORMAL_WIDTH/2

  if columnchart:
    width = NORMAL_WIDTH/maxbar
  else:
    width = NORMAL_WIDTH
  

  yshift = bar_groups_count.copy()
  yshift = {key: 0.0 for key in yshift}


  #Count bars and get new values
  current_bar_groups_count = bar_groups_count.copy()
  
  #Reset Dicts
  for key in current_bar_groups_count.keys():
    current_bar_groups_count[key] = 0.5
  for dat in data:
    #List for new_x with shift of next bars
    color = color_gen()
    index=0
    new_value_x: list = dat["x_data"].copy()
    new_value_y: list = dat["y_data"].copy()
    current_yshift: list = [0]*len(new_value_y)

    for current_num in new_value_x:
      key = str(current_num)
      if key in bar_groups_count.keys():
        #Shifts value depending if how much bars are already on the spot

        # only shift x Axis or offset the start height 
        if (columnchart):
          new_value_x[index] = xshift + current_bar_groups_count[str(key)]*width+current_num
        else:
          current_yshift[index] = yshift[key]
        current_bar_groups_count[str(key)] += 1
        yshift[key] += new_value_y[index]


      index+=1
    bars.append(axes.bar(new_value_x,new_value_y,color=color,label=dat["label"],width=width, bottom=current_yshift))
  return bars

def configure_plot()->list:
  global axes, config
  data: dict = config["data"]
  lines= []
  for dat in data:
    color = color_gen()
    lines.append(axes.plot(dat["x_data"],dat["y_data"],"-",color=color,label=dat["label"]))
  return lines

def configure_scatter()->list:
  global axes, config
  data: dict = config["data"]
  points = []
  for dat in data:
    color = color_gen()
    points.append(axes.scatter(dat["x_data"],dat["y_data"],color=color,label=dat["label"]))
  return points

def configure_pie()->list:
  global axes, config
  data: dict = config["data"]
  pie_pieces = []
  values = []
  colors = []
  labels = []

  for dat in data:
    colors.append(color_gen())

    val = 0
    for i in dat["y_data"]:
      val+=i
    values.append(val)
    
    labels.append(dat["label"])
  pie_pieces.append(axes.pie(values,labels=labels,colors=colors))
  return pie_pieces

#------Wrappers for config-------#

def handle_config(needs_title: bool = True,
                  needs_axes: bool = True,
                  needs_annotations: bool = True,
                  needs_styling = False, 
                  items: list = []):
  if needs_title:
    add_title()
  if needs_axes:
    add_axes_description()
  if needs_annotations:
    create_annotations(items)
  if items != []:
    if needs_styling:
      apply_shadow(items)


def map_annotation_to_data(annotation: dict,data_id: int,items :list):
  global config
  pos = (0.0,0.0)
  annotations = config["annotations"]
  for data_value,item in enumerate(items):
    #For Bars
    if isinstance(item,Rectangle):
      pos = item.get_center()
    #For Pie
    elif isinstance(item,Text):
      pos = item.get_position()
      pos = (pos[0]*PIE_ANNO_OFFSET,pos[1]*PIE_ANNO_OFFSET)
    #For Scatter
    elif isinstance(item,MaskedArray):
      pos = item.data.tolist()
    #For Line
    elif isinstance(item,tuple):
      pos = item
    else:
      return

    if annotation["data_id"] == data_id and annotation["data_value"] == data_value:
      annotation["x_value"] = pos[0]
      annotation["y_value"] = pos[1]


def create_annotations(items = []):
  global axes, config
  annotations = config["annotations"]
  number: int = 0
  ANNOTATION_TEXT_Y_OFFSET: float = -0.15
  ANNOTATION_TEXT_Y_SPACE: float = -0.05
  lb_offset=0




  for annotation in annotations:

    #Used for annotations that should be pinned on the items with a dataid
    handle_anno_without_fixed_pos(annotation,items)

    #Mapping x and y on chart
    x_val = annotation["x_value"]
    y_val = annotation["y_value"]
    txt:str = annotation["text"]
    if x_val != None and y_val != None and txt != None:
      number+=1
      y_txt = ANNOTATION_TEXT_Y_OFFSET+number*ANNOTATION_TEXT_Y_SPACE+lb_offset*ANNOTATION_TEXT_Y_SPACE

      plt.text(x_val,y_val, "(" + str(number) + ")",bbox=dict(boxstyle='circle,pad=0.1',facecolor='white',edgecolor='#0F1520',alpha=0.5))
      plt.text(0, y_txt,"(" + str(number) + ")" + txt,transform=axes.transAxes,ha='left', va='bottom')

    lb_offset += txt.count("\n")


def handle_anno_without_fixed_pos(annotation:dict, items:list):
    if annotation["data_id"] == None or annotation["data_value"] == None:
      return
    for data_id,item in enumerate(items):
      #For Scatter
      if isinstance(item,PathCollection):
        map_annotation_to_data(annotation,data_id,item.get_offsets())
      #For Bars
      if isinstance(item,BarContainer):
        map_annotation_to_data(annotation,data_id,item)
      #For Pie
      elif isinstance(item,tuple):
        map_annotation_to_data(annotation,data_id,item[1])
      #For Line
      elif isinstance(item,list) and len(item) > 0 and isinstance(item[0],Line2D):
        pos_list = list(zip(item[0].get_xdata(),item[0].get_ydata()))
        map_annotation_to_data(annotation,data_id,pos_list)



def add_axes_description():
    global axes, config
    plt.xlabel(config["xlabel"])
    plt.ylabel(config["ylabel"])
    axes.set(xlim=config["x_lim"], xticks=config["x_ticks"],ylim=config["y_lim"], yticks=config["y_ticks"])
    axes.set_xticklabels(config["x_tick_label"])
    axes.set_yticklabels(config["y_tick_label"])

def add_title():
    global config
    titlename = config["titlename"]
    plt.title(titlename)

#------Optional styling-------#

def init_style():
  style_params ={
    "figure.figsize": (12, 6),
    "figure.dpi": 120,
    "axes.grid": True,
    "axes.grid.axis": 'y',
    "axes.axisbelow": True,
    "font.size": "10",
    "axes.labelsize": 15,
    "axes.titlesize": 30,
    "axes.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "font.family": "Arial",
    "axes.titlelocation": "left",
}
  plt.rcParams.update(style_params)

def configure_style():
  global colorstack, axes
  colorstack = COLOR_STACK.copy()

  axes.spines["left"].set_linewidth(2)
  axes.spines["bottom"].set_linewidth(2)
  axes.spines["left"].set_alpha(0.4)
  axes.spines["bottom"].set_alpha(0.4)

  axes.spines["top"].set_visible(False)
  axes.spines["right"].set_visible(False)

  axes.patch.set_alpha(0.0)
  axes.set_facecolor("black")
  axes.patch.set_linewidth(15)

def apply_shadow(items):
  if items != None:
    for item in items:
      for i in item:
        i.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground="white"),
                path_effects.Normal()
          ])

def color_gen() -> str:
  global colorstack
  if len(colorstack) != 0:
    color = colorstack.pop()
  else:
    brightness = random.random()
    color = (brightness, brightness, 1.0)
  return color

#------Check if invalid symbols are used-------#

def check_input():
  check_input_for_invalid_symbols()
  check_input_for_format()

def check_input_for_invalid_symbols():
  clean_list = [
    "titlename",
    "xlabel",
    "ylabel",
    "x_tick_label",
    "y_tick_label"
  ]
  substrings = [
    "\n",
    "\t",
    "{"
    "}"
  ]
  erase_substring(clean_list,substrings)
  
  pass

def erase_substring(dict_keys_to_check: list[str], substrings: list[str]):
  global config
  for substring in substrings:
    for key in dict_keys_to_check:
      item = config[key]
      if type(item) is str:
        config[key] = item.replace(substring,"")
      elif type(item) is tuple:
        newlist = list(item)
        for i in range(len(newlist)):
          newlist[i] = newlist[i].replace(substring,"")
        config[key] = tuple(newlist)

def check_input_for_format():
  global config
  #config["titlename"] = check_input_for_lb(config["titlename"],TITLE_LENGTH_BEFORE_LB)
  for anno in config["annotations"]:
    anno["text"] = check_input_for_lb(anno["text"],ANNO_LENGTH_BEFORE_LB)

def check_input_for_lb(dict_value: str, length: int):
  global config
  if len(dict_value) > length:

    words: list[str] = dict_value.split()
    new_words: list[str] = []
    
    word_count = 0
    for word in words:
      new_word_count = word_count + len(word)
      if new_word_count > length:
        word = "\n"+word
        word_count = len(word)
      else:
        word_count = new_word_count

      new_words.append(word)
    return " ".join(new_words)
  return dict_value

#------Main function-------#

def generate_from_template(config_: dict):
    global axes, config
    config = copy.deepcopy(config_)
    check_input()
    #Style before getting axes
    init_style()
    #Getting axes and for for further styling
    fig,axes = plt.subplots()
    #Style for every chart
    configure_style()
    #Create Lines,bars etc
    handle_chart_type()
    #Show generated chart
    axes.legend()
    plt.show()