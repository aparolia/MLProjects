# Tree.py
# -------
# Ayush Parolia

class Tree:

    def __init__(self, value, level):
        self.value = value
        self.children = {}
        self.level = level

    def __str__(self):
        indent = ""
        for i in range(self.level):
            indent += "\t"
        string = "["+str(self.level)+"] Node:"+str(self.value)+"\n"
        for child in self.children.keys():
            string += indent+"\t|-"+str(child)+"-"+str(self.children[child])+"\n"
        return string

    def getValue(self):
        return self.value

    def addChild(self, subTree, decision):
        self.children[decision] = subTree

    def isLeaf(self):
        if len(self.children) == 0:
            return True
        return False
        
