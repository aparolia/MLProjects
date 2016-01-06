Name : Ayush Parolia

-- COMMAND to execute the code:
python id3.py -d <max_depth>

NOTE: We have only used maximum depth hyper-parameter so use only -d option, others will not have any effect on the decision tree

-- HELP in reading the output decision tree
Following is the structure of a node, printed on the screen

Root Node: 
[<Level>] Node: <Index of the attribute to choose/the label if its the only node>

Any other Node:
|-<decision value>-[<level>] Node: <Index of the attribute to choose/the label if its the leaf node>
