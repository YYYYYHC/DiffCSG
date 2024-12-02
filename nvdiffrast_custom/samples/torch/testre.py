import re

def replace_variable(expression):
    # 定义替换逻辑的函数
    def repl(match):
        variable_name = match.group(0)
        return f'variable["{variable_name}"]'
    
    # 使用正则表达式匹配所有变量并替换
    new_expression = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', repl, expression)
    return new_expression

expression = "-test1*2 -3 /test3"
new_expression = replace_variable(expression)
print(new_expression)
