from nltk.grammar import Nonterminal
from nltk.tree import Tree
import nltk
from lexer_analysis import *
import pandas as pd

grammar1 = nltk.CFG.fromstring("""
program -> 'START' statements 'END'

statements -> branch_stmt | 'EMP' assign1 | 'EMP' block| statements block 'EMP'| statements statements | statements assign1 | statements assign1 'EMP' | 'EMP' print_stmt 'newline' | 'EMP' input_stmt 'newline' | statements block 'newline' | statements assign1 'newline' | statements 'EMP' | stmt| statements 'newline' | end_branch_stmt | statements exp 'EMP' |statements exp 'newline' |'EMP' exp | 'EMP' ini 'EMP' | else_block 'EMP' | 'EMP' ini 'newline' | ifelse_stmt1 'EMP' | ifelse_stmt2 'EMP' | ifelse_stmt3 'EMP' | elif_stmt 'EMP' | input_stmt 'EMP' | state_input 'EMP' | statements for_stmt | statements while_stmt

branch_stmt -> 'EMP' if_stmt | 'EMP' while_stmt | 'EMP' for_stmt | 'EMP' ini_st 'EMP' | 'EMP' print_stmt 'EMP'

state_input -> 'EMP' input_stmt

end_branch_stmt -> statements for_stmt 'EMP' | statements while_stmt 'EMP' | statements if_stmt 'EMP' | assign 'EMP' | inp 'EMP' | inp1 'EMP'

ifelse_stmt1 -> if_stmt elif_stmt else_block
ifelse_stmt3 -> if_stmt else_block

exp_stmt1 -> number '*' identifer 'semi-colon' | number '-' identifer 'semi-colon' | number '+' identifer 'semi-colon' | number '/' identifer 'semi-colon' | identifer '*' number 'semi-colon' | identifer '-' number 'semi-colon' | identifer '+' number 'semi-colon' | identifer '/' number 'semi-colon'

exp_stmt -> number '*' number 'semi-colon' | number '-' number 'semi-colon' | number '+' number 'semi-colon' | number '/' number 'semi-colon' | identifer '*' identifer 'semi-colon' | identifer '-' identifer 'semi-colon' | identifer '+' identifer 'semi-colon' | identifer '/' identifer 'semi-colon' | exp_stmt1 

assign -> identifer '=' exp_stmt | identifer '=' 'string' | identifer '=' number "newline" | identifer '=' number "EMP" | identifer '=' bool_value | identifer '=' exp | identifer '=' '(' exp_stmt ')' | assign assign | i_r | identifer '=' identifer 'newline' | identifer '=' identifer 'EMP' | assign 'newline' assign

assign1 -> specifier assign

ini_st -> ini 'newline' assign 'semi-colon'

i_r -> i_r1 'semi-colon'

i_r1 -> identifer '+' '=' number | identifer '+' '+' | identifer '-' '=' number | identifer '-' '-' | identifer '*' '=' number | identifer '/' '=' number | identifer '+' '=' identifer | identifer '-' '=' identifer | identifer '*' '=' identifer | identifer '/' '=' identifer

exp -> identifer compare identifer | identifer compare number | number compare number | bool_value | identifer compare '(' exp_stmt ')'

for_stmt -> inp 'newline' 'for' '(' ini exp 'semi-colon' i_r1 ')' 'open-curly-bracket' 'newline' assign 'newline' 'close-curly-bracket' | inp 'newline' 'for' '(' ini exp 'semi-colon' i_r1 ')' 'open-curly-bracket' 'newline' inp 'newline' 'close-curly-bracket' | 'for' '(' ini exp 'semi-colon' i_r1 ')' 'open-curly-bracket' 'newline' statements | 'for' '(' ini exp 'semi-colon' i_r1 ')' 'open-curly-bracket' 'newline' inp 'newline' 'close-curly-bracket' | 'for' '(' ini exp 'semi-colon' i_r1 ')' 'open-curly-bracket' 'newline' assign 'newline' 'close-curly-bracket' | 'for' '(' ini exp 'semi-colon' i_r1 ')' 'open-curly-bracket' 'newline' if_stmt 'newline' 'close-curly-bracket' | 'for' '(' ini exp 'semi-colon' i_r1 ')' 'open-curly-bracket' 'newline' ifelse_stmt1 'newline' 'close-curly-bracket' | 'for' '(' ini exp 'semi-colon' i_r1 ')' 'open-curly-bracket' 'newline' ifelse_stmt2 'newline' 'close-curly-bracket' | 'for' '(' ini exp 'semi-colon' i_r1 ')' 'open-curly-bracket' 'newline' ifelse_stmt3 'newline' 'close-curly-bracket'

if_stmt -> 'if' '(' exp ')' 'open-curly-bracket' 'newline' assign 'newline' 'close-curly-bracket' | 'if' '(' exp ')' 'open-curly-bracket' 'newline' assign 'newline' 'close-curly-bracket' | 'if' '(' exp ')' 'open-curly-bracket' 'newline' inp 'newline' 'close-curly-bracket'

else_block -> 'else' 'open-curly-bracket' 'newline' assign 'newline' 'close-curly-bracket' | 'else' 'open-curly-bracket' 'newline' inp 'newline' 'close-curly-bracket'

elif_stmt -> 'else' if_stmt

while_stmt -> inp 'newline' 'while' '(' exp ')' 'open-curly-bracket' 'newline' assign 'newline' 'close-curly-bracket' | inp 'newline' 'while' '(' exp ')' 'open-curly-bracket' 'newline' inp 'newline' 'close-curly-bracket' | 'while' '(' exp ')' 'open-curly-bracket' 'newline' statements | 'while' '(' exp ')' 'open-curly-bracket' 'newline' inp 'newline' 'close-curly-bracket' | 'while' '(' exp ')' 'open-curly-bracket' 'newline' assign 'newline' 'close-curly-bracket' | 'while' '(' exp ')' 'open-curly-bracket' 'newline' if_stmt 'newline' 'close-curly-bracket' | 'while' '(' exp ')' 'open-curly-bracket' 'newline' ifelse_stmt1 'newline' 'close-curly-bracket' | 'while' '(' exp ')' 'open-curly-bracket' 'newline' ifelse_stmt2 'newline' 'close-curly-bracket' | 'while' '(' exp ')' 'open-curly-bracket' 'newline' ifelse_stmt3 'newline' 'close-curly-bracket'

input_stmt -> 'scanf' '(' 'quotes' type 'quotes' 'comma' 'and' identifer ')' 'semi-colon'

print_stmt -> 'printf' '(' 'quotes' type 'quotes' 'comma' identifer ')' 'semi-colon' | 'printf' '(' 'quotes' type 'quotes' 'comma' i_r1 ')' 'semi-colon' | 'printf' '(' 'quotes' type 'quotes' 'comma' number ')' 'semi-colon' | 'printf' '(' 'quotes' 'string' 'quotes' ')' 'semi-colon'

inp -> input_stmt | print_stmt
inp1 -> inp 'newline' inp

identifer -> 'var' | '(' 'var' ')' | identifer 'comma' identifer

number -> '(' number ')' | 'num'

compare -> '<'|'>'|'>='|'<='|'=='

bool_value -> '1' | '0'

specifier -> 'int' | 'float' | 'double' | 'char' | 'long long'

type -> 'percent' 'd' | 'percent' 'f' | 'percent' 'c'

ini -> specifier identifer 'semi-colon' | specifier identifer '=' number 'semi-colon' | specifier identifer '=' identifer 'semi-colon' | specifier identifer 'semi-colon'


""")

def convert_to_sym(list_sym):
    data=[]
    for i in list_sym:
        try:
            data.append(i.label())
        except:
            data.append(i)
    return data

def convert_to_action(data):
    try:
        action=f"reduce: {data.productions()[0]}"
    except:
        action="shift"
    return action

def shift_reduce_table(history,list_reduce,syntax):
    table={
        "stack":[],
        "curr_sym":[],
        "rest_of_input":[],
        "action":[]
    }
    limit=len(history)
    for i in range(limit):
        table["stack"].append(convert_to_sym(history[i][0]))
        table["curr_sym"].append(history[i][1][0] if len(history[i][1])>1 else "")
        table["rest_of_input"].append(history[i][1][1:])
        if i+1<limit:
            value=convert_to_action(history[i+1][0][-1])
            table["action"].append(value)
        else:
            table["action"].append("Accept" if syntax else "Fail")
    return table

class SteppingShiftReduceParser(nltk.SteppingShiftReduceParser):
    def __init__(self,grammar1,trace=2):
        super().__init__(grammar1,trace)
        self.table= {
                        "stack":[[]],
                        "curr_sym":[],
                        "rest_of_input":[],
                        "action":[]
                    }
    def _trace_stack(self, stack, remaining_text, marker=" "):
        s = "  " + marker + " [ "
        if marker=="S":
            self.table["action"].append("shift")
        elif marker=="R":
            if isinstance(stack[-1], Tree):
                self.table["action"].append("reduce: "+repr(Nonterminal(stack[-1].productions()[0])))
        curr_sym=None
        curr_stack=[]
        rest_of_input=remaining_text[-1::-1]
        for elt in stack:
            if isinstance(elt, Tree):
                curr_sym_=str(Nonterminal(elt.label()))
                s += repr(curr_sym_) + " "
                curr_stack.append(str(curr_sym_))
            else:
                curr_sym=str(elt)
                s += repr(curr_sym) + " "
                curr_stack.append(str(curr_sym))
        if marker =="S":
            self.table["curr_sym"].append(curr_sym)
        else:
            self.table["curr_sym"].append(rest_of_input.pop() if len(remaining_text)>0 else "")
        self.table["rest_of_input"].append(rest_of_input[-1::-1])
        self.table["stack"].append(curr_stack.copy())
        s += "* " + " ".join(remaining_text) + "]"
        print(s)

def print_table(table):
    if table["stack"][-1]==["program"]:
        table["curr_sym"].append("")
        table["rest_of_input"].append([])
        table["action"].append("Accept")
    else:
      table["curr_sym"].append("")
      table["rest_of_input"].append([])
      table["action"].append("Fail")
    new_data={}
    for i in table.keys():
        if i in ["action","curr_sym"]:
            new_data.update({i.upper():table[i]})
            continue
        new_data.update({i.upper():[]})
        for j in table[i]:
            new_data[i.upper()].append(" ".join(j))
    df=pd.DataFrame(new_data)
    
    return df


def checkSyntax(c):
    sent = c
    syntax = False
    rd_parser = SteppingShiftReduceParser(grammar1)
    tree=list(rd_parser.parse(sent))    
    history=rd_parser.table
    try:
        tree=tree[0]
        if tree.label()=="program":
            syntax = True
    except:
        #print(tree)
        pass
    return (syntax,print_table(history))

