import re
from graphviz import Digraph

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def tokenize(expression):
    # 匹配多字符变量名、数字和操作符
    return re.findall(r"\d+|[a-zA-Z0-9]+|\+|\-|\*|\/|\(|\)", expression)

def create_node(op, right, left):
    node = TreeNode(op)
    node.left = left
    node.right = right
    return node

def build_expression_tree(infix_tokens):
    ops = []  # 操作符栈
    values = []  # 操作数栈

    for token in infix_tokens:
        if token.isalnum():  # 如果是操作数
            values.append(TreeNode(token))
        elif token in "+-*/":  # 如果是操作符
            ops.append(token)
        elif token == '(':
            continue  # 对于这个简化的场景，忽略左括号
        elif token == ')':
            # 每遇到一个右括号，就处理一个操作符和两个操作数
            op = ops.pop()
            right = values.pop()
            left = values.pop()
            values.append(create_node(op, right, left))

    return values.pop()  # 表达式的根节点

def visualize_tree(node, graph=None):
    if graph is None:
        graph = Digraph()
        graph.node(name=str(id(node)), label=str(node.value))

    if node.left:
        graph.node(name=str(id(node.left)), label=str(node.left.value))
        graph.edge(str(id(node)), str(id(node.left)))
        visualize_tree(node.left, graph)

    if node.right:
        graph.node(name=str(id(node.right)), label=str(node.right.value))
        graph.edge(str(id(node)), str(id(node.right)))
        visualize_tree(node.right, graph)

    return graph
# 示例表达式
expression = "(((ab1-cd2)-ef3)+gh4)"
tokens = tokenize(expression)
root = build_expression_tree(tokens)
graph = visualize_tree(root)
graph.render(filename='//home/cli7/CSGDR/nvdiffrast/CodeBase/expression_tree', format='pdf')
