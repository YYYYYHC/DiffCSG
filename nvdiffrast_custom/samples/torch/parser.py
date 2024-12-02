import pdb
import re
import torch
# from graphviz import Digraph
# Adjust the approach for handling the tree structure and ensure correct parsing and building
class TreeNode:
    def __init__(self, type, name, data=None):
        self.type = type  # "operator" or "primitive"
        self.name = name
        self.data = data
        self.render_data = None
        self.left = None
        self.right = None
def is_numeric(s):
    try:
        float(s)  # 尝试将字符串s转换为浮点数
        return True
    except ValueError:
        return False
def is_numeric_expression(s):
    import operator
    # 创建一个安全的环境，只包含数学运算符
    allowed_operators = {
        '__builtins__': None,  # 不允许访问任何内置函数和变量
        'abs': abs,
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv,
        'pow': pow
    }
    
    try:
        # 尝试评估字符串s作为表达式
        eval(s, {"__builtins__": None}, allowed_operators)
        return True
    except:
        return False
#########
def tokenize(expression):
    # 匹配多字符变量名、数字和操作符
    return re.findall(r"\d+|[a-zA-Z0-9]+|\+|\-|\*|\/|\(|\)", expression)
def create_node(op, right, left):
    op.left = left
    op.right = right
    return op

def build_expression_tree(infix_tokens, nodes):
    ops = []  # 操作符栈
    values = []  # 操作数栈
    
    for token in infix_tokens:
        if token.isalnum():  # 如果是操作数
            values.append(nodes[token])
        elif token in "+-*/":  # 如果是操作符
            ops.append(TreeNode('operator', token))
        elif token == '(':
            continue  # 对于这个简化的场景，忽略左括号
        elif token == ')':
            # 每遇到一个右括号，就处理一个操作符和两个操作数
            if len(ops) == 0:
                op = values.pop()
                right = None
                left = None
            else:
                op = ops.pop()
                right = values.pop()
                left = values.pop()
            values.append(create_node(op, right, left))

    return values.pop()  # 表达式的根节点

def replace_variable(expression):
    # 定义替换逻辑的函数
    def repl(match):
        variable_name = match.group(0)
        return f"variables['{variable_name}']"
    
    # 使用正则表达式匹配所有变量并替换
    new_expression = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', repl, expression)
    return new_expression
# Use the previously corrected primitive data parsing function
def parse_primitive_data_final(line, variables=None, require_grad = False, all_param=True):
    
    parts = line.split(':')
    name = parts[0].strip()
    data_str = ':'.join(parts[1:]).strip().strip('[]')  # Handle potential ':' within data correctly
    data = []
    for x in data_str.split(','):
           
        x = x.strip()
        
        if  variables is not None and not is_numeric_expression(x):
            if all_param:
                data.append(torch.tensor(eval(replace_variable(x)), dtype=torch.float32).cuda())
            else:
                data.append(replace_variable(x))
        else :
            data.append(torch.tensor(eval(x), dtype=torch.float32).cuda())

    return TreeNode("primitive", name, data)

def print_tree(node, level=0):
    indent = "  " * level
    print(f"{indent}{node.type}, {node.name}, {node.data if node.data else 'None'}")
    print_tree(node.left, level + 1) if node.left else None
    print_tree(node.right, level + 1) if node.right else None
# Adjust the tree building logic to correctly handle the operator and child nodes


def build_tree_from_file_properly(file_path, variables=None, require_grad = False, all_param=True):
    nodes = {}
    with open(file_path, 'r') as file:
        for line in file:
            
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            if line.startswith("cube") or line.startswith("cylinder") or line.startswith("polyline") or line.startswith("sphere"):
                node = parse_primitive_data_final(line, variables, require_grad, all_param)
                nodes[node.name] = node
            elif line.startswith("Tree"):
                tree_structure = line.split(':')[1].strip()
    tree_structure_token = tokenize(tree_structure)
    tree_root = build_expression_tree(tree_structure_token, nodes)
    # pdb.set_trace()
    # # Assuming the file structure is known and fixed for simplicity
    # operator_root = TreeNode("operator", "OP_sub")
    # operator_root.add_child(nodes["Cube2"])
    # operator_root.add_child(nodes["Cube1"])

    return tree_root

def visualize_tree(node, graph=None):
    if graph is None:
        graph = Digraph()
        graph.node(name=str(id(node)), label=str(node.name))

    if node.left:
        graph.node(name=str(id(node.left)), label=f"{node.left.name} : {node.left.data}")
        graph.edge(str(id(node)), str(id(node.left)))
        visualize_tree(node.left, graph)

    if node.right:
        graph.node(name=str(id(node.right)), label=f"{node.right.name} : {node.right.data}")
        graph.edge(str(id(node)), str(id(node.right)))
        visualize_tree(node.right, graph)

    return graph


# 示例表达式
# expression = "(((ab1-cd2)-ef3)+gh4)"
# tokens = tokenize(expression)
# root = build_expression_tree(tokens)
# graph = visualize_tree(root)
# graph.render(filename='//home/cli7/CSGDR/nvdiffrast/CodeBase/expression_tree', format='pdf')
if __name__ == "__main__":
    file_path = "/home/cli7/CSGDR/nvdiffrast/CodeBase/twoCubeSub.txt"
    tree_root = build_tree_from_file_properly(file_path)
    print_tree(tree_root)
    graph = visualize_tree(tree_root)
    graph.render(filename='/home/cli7/CSGDR/nvdiffrast/CodeBase/expression_tree', format='pdf')
