
import os
import psycopg2
import psycopg2.extras
import tabulate
import re
import ast
import pandas as pd
from dotenv import load_dotenv

SQL_PRECEDENCE = {
    ast.Or: 1,       
    ast.And: 2,      
    ast.Not: 3,      
    ast.Lt: 4,
    ast.Gt: 4,
    ast.Eq: 4,
    ast.NotEq: 4,
    ast.LtE: 4,
    ast.GtE: 4,
}
OP_SYMBOLS = {
    ast.Or: "or",
    ast.And: "and",
    ast.Not: "not",
    ast.Lt: "<",
    ast.Gt: ">",
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.LtE: "<=",
    ast.GtE: ">=",
    ast.Mult: "*",    
    ast.Div: "/",     
    ast.Add: "+",     
    ast.Sub: "-",     
    ast.Mod: "%",
}

def parse_condition_sql(condition):
    tree = ast.parse(condition, mode="eval")
    def format_node(node):
        if isinstance(node, ast.BoolOp):
            op = OP_SYMBOLS[type(node.op)]
            args = sorted([format_node(value) for value in node.values], key=lambda x: SQL_PRECEDENCE[type(node.op)])
            return f"({f' {op} '.join(args)})"
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return f"NOT {format_node(node.operand)}"
        elif isinstance(node, ast.Compare):
            left = format_node(node.left)
            comparisons = " ".join(f"{OP_SYMBOLS[type(op)]} {format_node(comp)}" for op, comp in zip(node.ops, node.comparators))
            return f"({left} {comparisons})"
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return f"'{node.value}'"
            return str(node.value)
        elif isinstance(node, ast.BinOp):
            left = format_node(node.left)
            right = format_node(node.right)
            op = OP_SYMBOLS[type(node.op)]
            return f"({left} {op} {right})"
        else:
            raise ValueError(f"Unsupported node type: {type(node).__name__}")
    return format_node(tree.body)

def preprocess_expression(expression):
    expression = expression.replace('’', "'").replace('‘', "'")
    return expression

def rearrange_variable_names(expression):
    expression = preprocess_expression(expression)
    expression = re.sub(r'\b_?([a-zA-Z_]*)(\d+)(\w*\.?\w*)?\b', r'\1\3\2', expression)
    expression = re.sub(r'\.', '_', expression)
    expression = re.sub(r'\b_+', '', expression)
    expression = re.sub(r'(?<!\s)([<>])(?!=)', r' \1 ', expression)
    expression = re.sub(r'(?<!\s)=(?!=)', r'==', expression)
    return expression.strip()

def process_txt(lines):
    lines = lines.splitlines()
    emf_struct = {}
    for i, line in enumerate(lines):
        if ":" in line:
            i1, i2 = line.find("("), line.find(")")
            var = line[i1+1:i2]
            try:
                ib1 = var.index("[")
                ib2 = var.index("]")
                var = var[ib1+1:ib2]
            except:
                pass
            i = i + 1
            if var == "n":
                emf_struct[var] = lines[i]
            elif var == "G":
                vars = rearrange_variable_names(lines[i])
                emf_struct[var] = parse_condition_sql(vars)
            else:
                emf_struct[var] = []
                while ":" not in lines[i]:
                    if var == "σ":
                        vars = rearrange_variable_names(lines[i])
                        emf_struct[var].append(parse_condition_sql(vars))
                    else:
                        if "," in lines[i]:
                            splitting_var = ","
                        else:
                            emf_struct[var].append(rearrange_variable_names(lines[i]))
                        vars = rearrange_variable_names(lines[i])
                        vars = vars.split(splitting_var)
                        vars = [v.strip() for v in vars]
                        emf_struct[var] = vars
                    i = i + 1
                    if(i == len(lines)):
                        break
    for i, where in enumerate(emf_struct['σ']):
        emf_struct['σ'][i] = re.sub(r'\b\w+\b', replace_with_emf_or_gb, where)
    print(emf_struct)
    return emf_struct

def make_df(table):
    df = pd.DataFrame(table)
    return df

def replace_with_emf_or_gb(match):
    variable = match.group(0)
    if re.match(r'^\w+_\w+\d+$', variable):
        return f"gb['{variable}']"
    elif re.match(r'^[A-Za-z_]+\d+$', variable):
        word_part = re.match(r'^([A-Za-z_]+)\d+$', variable).group(1)
        return f"row['{word_part}']"
    elif re.match(r'^\w+_\w+$', variable):
        return f"normal_aggregates['{variable}']"
    else:
        return variable
  

def make_emf_struct(emf_struct, df):
    gvs = df[emf_struct['V']].drop_duplicates()
    mf_struct = [{v: row[i] for i, v in enumerate(emf_struct['V'])} for row in gvs.values]
    print(mf_struct)
    normal_aggregates = {}
    for f in emf_struct['F']:
        if not any(char.isdigit() for char in f):
            match = re.search(r'(?<=_)(\D+)(?=\d*$)', f)
            val = match.group(1)
            if 'min' in f:
                normal_aggregates[f] = min(df[val]) 
            elif 'max' in f:
                normal_aggregates[f] = max(df[val]) 
            elif 'count' in f:
                normal_aggregates[f] = len(df)
            elif 'avg' in f:
                normal_aggregates[f] = sum(df[val])/len(df)
            elif 'sum' in f:
                normal_aggregates[f] = sum(df[val]) 
    for i, where in enumerate(emf_struct['σ']):
        f2 = []
        for f in emf_struct['F']: 
            match = re.search(r'(?<=_)(\D+)(?=\d*$)', f) 
            val = match.group(1)
            if str(i+1) in f:
                f2.append(f)
        for j, row in df.iterrows():
            for gb in mf_struct:
                state = gb.get('state')
                cust = gb.get('cust')
                day = gb.get('day')
                month = gb.get('month')
                prod = gb.get('prod')
                year = gb.get('year')
                quant = gb.get('quant')
                date = gb.get('date')
                if eval(where):
                    for f in f2:
                        if 'min' in f:
                            try:
                                if gb[f] > row[val]:
                                    gb[f] = row[val]
                            except KeyError:
                                gb[f] = row[val] 
                        elif 'max' in f:
                            try:
                                if gb[f] < row[val]:
                                    gb[f] = row[val]
                            except KeyError:
                                gb[f] = row[val]
                        elif 'count' in f:
                            try:
                                gb[f] = gb[f] + 1
                            except KeyError:
                                gb[f] = 1
                        elif 'avg' in f:
                            try:
                                gb[f][1] = ((gb[f][1] * gb[f][0]) + row[val]) / (gb[f][0] + 1)
                                gb[f][0] = gb[f][0] + 1
                            except KeyError:
                                gb[f] = [1, row[val]]
                        elif 'sum' in f:
                            try:
                                gb[f] = gb[f] + row[val]
                            except KeyError:
                                gb[f] = row[val] 
    for gb in mf_struct:
        for f in emf_struct['F']:
            if isinstance(gb.get(f), list) and len(gb[f]) > 1:
                gb[f] = gb[f][1] 
    return mf_struct, normal_aggregates

def handle_having_conditions(emf_struct, mf_struct, normal_aggregates):
    having = emf_struct['G']
    modified_having = re.sub(r'\b\w+\b', replace_with_emf_or_gb, having)
    filtered_data = []
    for gb in mf_struct:
        if eval(modified_having):
            filtered_data.append(gb)
    return filtered_data

def handle_selection(emf_struct, mf_struct, normal_aggregates):
    for d in mf_struct:
        keys_to_remove = [key for key in d if key not in emf_struct['S']]
        for key in keys_to_remove:
            del d[key]
        for key in normal_aggregates:
            if(key in emf_struct['S']):
                d[key] = normal_aggregates[key]
    return mf_struct

# DO NOT EDIT THIS FILE, IT IS GENERATED BY generator.py

def query():
    load_dotenv()

    user = os.getenv('USER')
    password = os.getenv('PASSWORD')
    dbname = os.getenv('DBNAME')

    conn = psycopg2.connect("dbname="+dbname+" user="+user+" password="+password,
                            cursor_factory=psycopg2.extras.DictCursor)
    cur = conn.cursor()
    cur.execute("SELECT * FROM sales")
    lines = "SELECT ATTRIBUTE(S):\ncust, prod, 2_sum_quant, 2_avg_quant, 3_sum_quant, 3_avg_quant\nNUMBER OF GROUPING VARIABLES(n):\n4\nGROUPING ATTRIBUTES(V):\ncust, prod\nF-VECT([F]):\nsum_quant, 1_sum_quant, 1_avg_quant, 2_sum_quant, 2_avg_quant, 3_sum_quant, 3_avg_quant\nSELECT CONDITION-VECT([σ]):\n1.state='NY' and 1.cust=cust and 1.month=3\n2.state=’NJ’ and 2.cust=cust\n3.state=’CT’ and 3.cust=cust\nHAVING_CONDITION(G):\n3_avg_quant > 475"
    emf_struct = process_txt(lines)
    rows = cur.fetchall()
    column_names = [description[0] for description in cur.description]
    df = pd.DataFrame(rows, columns=column_names)
    mf_struct, normal_aggregates = make_emf_struct(emf_struct, df)
    filtered_mf_struct = handle_having_conditions(emf_struct, mf_struct, normal_aggregates)
    final_mf_struct = handle_selection(emf_struct, filtered_mf_struct, normal_aggregates)
    return tabulate.tabulate(final_mf_struct,
                        headers="keys", tablefmt="psql")

def main():
    print(query())
    
if "__main__" == __name__:
    main()
    