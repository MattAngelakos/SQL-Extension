"""
Names: Matthew Angelakos and Christopher Arias
Professor Kim
EMF Query Project
December 15 2024
Header Comment: 
Dependencies: Ensure sure that the psycopg2, dotenv, pandas, and postgresSQL are installed, The env file in the 
will need to be updated with whoever's credentials needed to connect to your database. If needed the sql file to create the 
sales table that is needed to run the program is included. Run load_sales_10000_table (NEW).sql to execute it.
Starting the Program: Type 'python generator.py' and the program will begin executing.
Running the Program: When the program is run, the terminal will prompt the user to input a file name or a manual input of the phi operatprs
inline. The input will then be processed and will generate an algorithm corresponding to the type of query specified by the
user input.
Output: The program will output to a file called '_generated.py', which can be run without input to return a table of the output
for the query. This process will run automatically after the generator.py is run however that can be changed by removing line 540 of this file.
"""
import os
import psycopg2
import subprocess
import psycopg2.extras
import tabulate
import re
import ast
import pandas as pd
from dotenv import load_dotenv

#These are the precedences of SQL so that we can build our abstract syntax tree for our select, where, and having conditions
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
#These are the symbols that we will be considering as inputs found in SQL but undercase per the prompt of the project
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

#function to parse the select, where, and having conditions of our query
def parse_condition_sql(condition):
    tree = ast.parse(condition, mode="eval") #initalize the tree
    def format_node(node): #this function will take in the root of the tree and recursively make our tree
        if isinstance(node, ast.BoolOp): #if the current node is a boolean operator we will execute this
            op = OP_SYMBOLS[type(node.op)] #get operator symbols
            args = sorted([format_node(value) for value in node.values], key=lambda x: SQL_PRECEDENCE[type(node.op)]) #sort precendence recurisvely
            return f"({f' {op} '.join(args)})" #return the operator
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not): #if the current node is a boolean operator we will execute this
            return f"NOT {format_node(node.operand)}"
        elif isinstance(node, ast.Compare): #if the current node is a comparison operator we will execute this
            left = format_node(node.left) #recursively generate the tree on the left hand side as the left of the comparison must be generated properly
            comparisons = " ".join(f"{OP_SYMBOLS[type(op)]} {format_node(comp)}" for op, comp in zip(node.ops, node.comparators)) #format the node per the precedence
            return f"({left} {comparisons})" #combined left recursion and comparisons just generated
        elif isinstance(node, ast.Name): #if variable just return
            return node.id
        elif isinstance(node, ast.Constant): #if a constant we make check if its a string(it should always be) if so return as such otherwise cast as a string and return
            if isinstance(node.value, str):
                return f"'{node.value}'"
            return str(node.value)
        elif isinstance(node, ast.BinOp): #similarly to the comparison for a binary operator we recurisvely get the left and right of which and combine it with the operator given
            left = format_node(node.left)
            right = format_node(node.right)
            op = OP_SYMBOLS[type(node.op)]
            return f"({left} {op} {right})"
        else:
            raise ValueError(f"Unsupported node type: {type(node).__name__}")
    return format_node(tree.body) #we return our tree by passing in the root


def preprocess_expression(expression): #‘ was giving us issues so we make them straight
    expression = expression.replace('’', "'").replace('‘', "'")
    return expression

def rearrange_variable_names(expression): #essentially this function just renames the input variables as the ones given are invalid in python
    expression = preprocess_expression(expression)
    expression = re.sub(r'\b_?([a-zA-Z_]*)(\d+)(\w*\.?\w*)?\b', r'\1\3\2', expression)
    expression = re.sub(r'\.', '_', expression)
    expression = re.sub(r'\b_+', '', expression)
    expression = re.sub(r'(?<!\s)([<>])(?!=)', r' \1 ', expression)
    expression = re.sub(r'(?<!\s)=(?!=)', r'==', expression)
    return expression.strip()

def process_txt(lines): #this function takes in a string(the query input) and processes it into a useable emf_struct of all 6 conditions
    lines = lines.splitlines() #make the string an array
    emf_struct = {} #initalize the structure(a dictionary in this case)
    for i, line in enumerate(lines): #obviously iterate through our lines, but we enumerate to have access to i 
        if ":" in line: #colon indicates a defining line for one of the 6 variables
            i1, i2 = line.find("("), line.find(")") #these upcoming lines identify which variable we are working on
            var = line[i1+1:i2] 
            try:
                ib1 = var.index("[")
                ib2 = var.index("]")
                var = var[ib1+1:ib2]
            except:
                pass
            i = i + 1 #we set our pointer to the next element of the array in order to extract the value of the variable key we just got
            if var == "n": #n is the simplest case its just the value of that line
                emf_struct[var] = lines[i]
            elif var == "G": #G is 2nd simplest case. First we must rearrange variable names then make our abstract syntax tree for that conditon then we can define it
                try:
                    vars = rearrange_variable_names(lines[i])
                    emf_struct[var] = parse_condition_sql(vars)
                except:
                    continue
            else: #our other cases can have more than 1 value defined as they are array values so a little more involved
                emf_struct[var] = [] #initalize this struct(an array)
                while ":" not in lines[i]: #keep going till we aren't at the next emf variable(only really important for σ)
                    if var == "σ": #same logic applies as having just with everyline in the conditionals
                        if len(lines[i]) > 0:
                            vars = rearrange_variable_names(lines[i])
                            emf_struct[var].append(parse_condition_sql(vars))
                    else: #otherwise we are making the select, F, or grouping attribues
                        if "," in lines[i]: #comma is the seperator
                            splitting_var = ","
                        else: #otherwise we have 1 and can just append
                            emf_struct[var].append(rearrange_variable_names(lines[i]))
                        vars = rearrange_variable_names(lines[i]) #otherwise append and iterate through each individual comma
                        vars = vars.split(splitting_var)
                        vars = [v.strip() for v in vars]
                        emf_struct[var] = vars
                    i = i + 1 #go to the next line
                    if(i == len(lines)): #if at the end then stop
                        break
    for i, where in enumerate(emf_struct['σ']):
        emf_struct['σ'][i] = re.sub(r'\\b\w+\\b', replace_with_emf_or_gb, where)
    return emf_struct #return the structre we just created

def make_df(table): #make the table we get for our sql table a dataframe as they are easier to work with
    df = pd.DataFrame(table)
    return df

def replace_with_emf_or_gb(match): #2nd regex, this will basically rename the variables in the emf struct for the sigma and having clauses to pull the data from the right sources
    variable = match.group(0)
    if re.match(r'^\w+_\w+\d+$', variable):  # Check for gb case
        if "avg" in variable: 
            return f"gb['{variable}'][0]"
        else:
            return f"gb['{variable}']"
    elif re.match(r'^[A-Za-z_]+\d+$', variable):
        word_part = re.match(r'^([A-Za-z_]+)\d+$', variable).group(1)
        return f"row['{word_part}']"
    elif re.match(r'^\w+_\w+$', variable):
        return f"normal_aggregates['{variable}']"
    else:
        return variable

def make_emf_struct(emf_struct, df): #this is the function to create the intial table of the mf_struct before the select or having clauses
    try:
        gvs = df[emf_struct['V']].drop_duplicates() #This is the first pass by isolating all distinct values of V
        mf_struct = [{v: row[i] for i, v in enumerate(emf_struct['V'])} for row in gvs.values] #initalize our struct to what was found and make it a dictionary for a table eventually
        handleV = True
    except:
        mf_struct = [{v: row[i] for i, v in enumerate(df.columns)} for row in df.values]
        handleV = False
    normal_aggregates = {} #we may need the normal aggregates depending on what the query calls for
    for f in emf_struct['F']: #iterate through the functions list
        if len(f) > 0:
            if not any(char.isdigit() for char in f): #check if no digits ie GV are present
                match = re.search(r'(?<=_)(\D+)(?=\d*$)', f) #match which value of the sales table it is
                val = match.group(1)
                if 'min' in f: #these if just compute the normal aggregates for whatever was found
                    normal_aggregates[f] = min(df[val]) 
                elif 'max' in f:
                    normal_aggregates[f] = max(df[val]) 
                elif 'count' in f:
                    normal_aggregates[f] = len(df)
                elif 'avg' in f:
                    normal_aggregates[f] = sum(df[val])/len(df)
                elif 'sum' in f:
                    normal_aggregates[f] = sum(df[val]) 
    for i, where in enumerate(emf_struct['σ']): #General loop that represents the EMF algo that will scan through the table loop through the GV first
        f2 = []
        for f in emf_struct['F']: #now we iterate through the aggregate functions
            match = re.search(r'(?<=_)(\D+)(?=\d*$)', f) #now like normal we do it for the current index
            val = match.group(1)
            if str(i+1) in f: #if the current GV is contained in the aggregate we use it (move line)
                f2.append(f)
        for j, row in df.iterrows(): #loop through the table
            for gb in mf_struct: #loop through the distinct group by values
                state = gb.get('state') #these will get the current valies of the row in order to compute the select condition
                cust = gb.get('cust')
                day = gb.get('day')
                month = gb.get('month')
                prod = gb.get('prod')
                year = gb.get('year')
                quant = gb.get('quant')
                date = gb.get('date')
                if eval(where): #Because the where is an AST with all the variables renamed we can directly evalute the select condition to determine if it matches the select condition
                    for f in f2: #now we iterate through the aggregate functions
                        if 'min' in f: #this is essentially done like before but for average we need a tuple and need to account for key errors(honestly fix whole function can be done easier)
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
    if handleV:
        for gb in mf_struct: #remove the tuple part of the averages afterwards
            for f in emf_struct['F']:
                if isinstance(gb.get(f), list) and len(gb[f]) > 1:
                    gb[f] = gb[f][1] 
    return mf_struct, normal_aggregates #return our final mf_struct and the normal aggregates

def handle_having_conditions(emf_struct, mf_struct, normal_aggregates):#this will apply our having to all rows of our mf_struct using the same logic of the select conditions
    try:
        having = emf_struct['G']
        modified_having = re.sub(r'\b\w+\b', replace_with_emf_or_gb, having)
        filtered_data = []
        for gb in mf_struct:
            if eval(modified_having):
                filtered_data.append(gb)
        return filtered_data
    except:
        return mf_struct

def handle_selection(emf_struct, mf_struct, normal_aggregates):#this function gets our final table ready if a column is in S keep it, otherwise remove
    for d in mf_struct:
        keys_to_remove = [key for key in d if key not in emf_struct['S']]
        for key in keys_to_remove:
            del d[key]
        for key in normal_aggregates: #move eventually
            if(key in emf_struct['S']):
                d[key] = normal_aggregates[key]
    return mf_struct

def main(): 
    """
    This is the generator code. It should take in the MF structure and generate the code
    needed to run the query. That generated code should be saved to a 
    file (e.g. _generated.py) and then run.
    """
    choice = input("Process Text File(1) or Manual Input(2):")#to make our file we use a choice of a manual or textfile inout
    if choice == '1': 
        txtfilename = input("Enter textfile name:")#if its the textfile we simply just open the file and read it
        f = open(txtfilename, "r")
        sql_string = f.read()
    elif choice == '2':#otherwise we go through each line in order to make all of our conditions met
        select_attributes = input("Enter SELECT ATTRIBUTE(S), separated by commas:\n").split(", ")
        num_grouping_vars = input("Enter NUMBER OF GROUPING VARIABLES(n):\n")
        grouping_attributes = input("Enter GROUPING ATTRIBUTES(V), separated by commas:\n").split(", ")
        f_vect = input("Enter F-VECT([F]), separated by commas:\n").split(", ")
        select_conditions = []
        for i in range(int(num_grouping_vars)):
            condition = input(f"Enter SELECT CONDITION-VECT([σ]) for condition {i + 1}:\n")
            select_conditions.append(condition)
        having_conditions = input("Enter HAVING_CONDITION(G):\n")
        sql_string = ( #combine the string together
            f"SELECT ATTRIBUTE(S):\n{', '.join(select_attributes)}\n"
            f"NUMBER OF GROUPING VARIABLES(n):\n{num_grouping_vars}\n"
            f"GROUPING ATTRIBUTES(V):\n{', '.join(grouping_attributes)}\n"
            f"F-VECT([F]):\n{', '.join(f_vect)}\n"
            "SELECT CONDITION-VECT([σ]):\n" +
            "\n".join(select_conditions) +
            f"\nHAVING_CONDITION(G):\n{having_conditions}"
        )
    else:
        print("invalid input")
    # Note: The f allows formatting with variables.
    #       Also, note the indentation is preserved.
    #all we are doing is literally just inputting all the functions above but with the query, literally the only thing that will change is that string
    tmp = f"""
import os
import psycopg2
import psycopg2.extras
import tabulate
import re
import ast
import pandas as pd
from dotenv import load_dotenv

SQL_PRECEDENCE = {{
    ast.Or: 1,       
    ast.And: 2,      
    ast.Not: 3,      
    ast.Lt: 4,
    ast.Gt: 4,
    ast.Eq: 4,
    ast.NotEq: 4,
    ast.LtE: 4,
    ast.GtE: 4,
}}
OP_SYMBOLS = {{
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
}}

def parse_condition_sql(condition):
    tree = ast.parse(condition, mode="eval")
    def format_node(node):
        if isinstance(node, ast.BoolOp):
            op = OP_SYMBOLS[type(node.op)]
            args = sorted([format_node(value) for value in node.values], key=lambda x: SQL_PRECEDENCE[type(node.op)])
            return f"({{f' {{op}} '.join(args)}})"
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return f"NOT {{format_node(node.operand)}}"
        elif isinstance(node, ast.Compare):
            left = format_node(node.left)
            comparisons = " ".join(f"{{OP_SYMBOLS[type(op)]}} {{format_node(comp)}}" for op, comp in zip(node.ops, node.comparators))
            return f"({{left}} {{comparisons}})"
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return f"'{{node.value}}'"
            return str(node.value)
        elif isinstance(node, ast.BinOp):
            left = format_node(node.left)
            right = format_node(node.right)
            op = OP_SYMBOLS[type(node.op)]
            return f"({{left}} {{op}} {{right}})"
        else:
            raise ValueError(f"Unsupported node type: {{type(node).__name__}}")
    return format_node(tree.body)

def preprocess_expression(expression):
    expression = expression.replace('’', "'").replace('‘', "'")
    return expression

def rearrange_variable_names(expression):
    expression = preprocess_expression(expression)
    expression = re.sub(r'\\b_?([a-zA-Z_]*)(\d+)(\w*\.?\w*)?\\b', r'\\1\\3\\2', expression)
    expression = re.sub(r'\.', '_', expression)
    expression = re.sub(r'\\b_+', '', expression)
    expression = re.sub(r'(?<!\s)([<>])(?!=)', r' \\1 ', expression)
    expression = re.sub(r'(?<!\s)=(?!=)', r'==', expression)
    return expression.strip()

def process_txt(lines):
    lines = lines.splitlines()
    emf_struct = {{}}
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
                try:
                    vars = rearrange_variable_names(lines[i])
                    emf_struct[var] = parse_condition_sql(vars)
                except:
                    continue
            else:
                emf_struct[var] = []
                while ":" not in lines[i]:
                    if var == "σ":
                        if len(lines[i]) > 0:
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
        emf_struct['σ'][i] = re.sub(r'\\b\w+\\b', replace_with_emf_or_gb, where)
    print(emf_struct)
    return emf_struct

def make_df(table):
    df = pd.DataFrame(table)
    return df

def replace_with_emf_or_gb(match): #2nd regex, this will basically rename the variables in the emf struct for the sigma and having clauses to pull the data from the right sources
    variable = match.group(0)
    if re.match(r'^\w+_\w+\d+$', variable):  # Check for gb case
        if "avg" in variable: 
            return f"gb['{{variable}}'][0]"
        else:
            return f"gb['{{variable}}']"
    elif re.match(r'^[A-Za-z_]+\d+$', variable):
        word_part = re.match(r'^([A-Za-z_]+)\d+$', variable).group(1)
        return f"row['{{word_part}}']"
    elif re.match(r'^\w+_\w+$', variable):
        return f"normal_aggregates['{{variable}}']"
    else:
        return variable
  

def make_emf_struct(emf_struct, df):
    try:
        gvs = df[emf_struct['V']].drop_duplicates()
        mf_struct = [{{v: row[i] for i, v in enumerate(emf_struct['V'])}} for row in gvs.values]
        handleV = True
    except:
        mf_struct = [{{v: row[i] for i, v in enumerate(df.columns)}} for row in df.values]
        handleV = False
    print(mf_struct)
    normal_aggregates = {{}}
    for f in emf_struct['F']:
        if len(f) > 0:
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
            if len(f) > 0: 
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
    if handleV:
        for gb in mf_struct:
            for f in emf_struct['F']:
                if isinstance(gb.get(f), list) and len(gb[f]) > 1:
                    gb[f] = gb[f][1] 
    print(mf_struct)
    return mf_struct, normal_aggregates

def handle_having_conditions(emf_struct, mf_struct, normal_aggregates):
    try:
        having = emf_struct['G']
        modified_having = re.sub(r'\\b\w+\\b', replace_with_emf_or_gb, having)
        filtered_data = []
        for gb in mf_struct:
            if eval(modified_having):
                filtered_data.append(gb)
        return filtered_data
    except:
        return mf_struct

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
    lines = {repr(sql_string)}
    emf_struct = process_txt(lines)
    rows = cur.fetchall()
    column_names = [description[0] for description in cur.description]
    df = pd.DataFrame(rows, columns=column_names)
    mf_struct, normal_aggregates = make_emf_struct(emf_struct, df)
    filtered_mf_struct = handle_having_conditions(emf_struct, mf_struct, normal_aggregates)
    final_mf_struct = handle_selection(emf_struct, filtered_mf_struct, normal_aggregates)
    result_df = pd.DataFrame(final_mf_struct)
    result_df.to_csv("output.csv", index=False)  # Set index=False to avoid saving index as a column
    return tabulate.tabulate(final_mf_struct,
                        headers="keys", tablefmt="psql")

def main():
    print(query())
    
if "__main__" == __name__:
    main()
    """
    # Write the generated code to a file
    open("_generated.py", "w").write(tmp)
    # Execute the generated code
    subprocess.run(["python3", "_generated.py"])


if "__main__" == __name__:
    #process_txt("test.txt")
    main()
