import csv
from neo4j import GraphDatabase #, data
from time import time, gmtime, strftime
import pprint
from optparse import OptionParser, OptionGroup
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from collections import Counter
import regex
import re
import logging
import spacy

import textacy
from textacy import text_stats as ts
import textacy.tm

import os.path
import networkx as nx
import matplotlib.pyplot as plt

import psycopg2
import pandas.io.sql as sqlio
import spacy

from collections import Counter
#import pygments
from pygments import lexers, token
from pygments.util import ClassNotFound

tool_version = "0.0.1.0002"

run = strftime("%Y%m%d%H%M%S_run_", gmtime())
run_log_file = os.path.abspath(os.path.join(os.getcwd(), run+"moduleBuilder.log"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(run_log_file),
        logging.StreamHandler()
    ]
)

pp = pprint.PrettyPrinter(indent=4)


# arguments

def arg_list_callback(option, opt, value, parser):
    if ',' in value:
        setattr(parser.values, option.dest, value.split(','))
    else:
        setattr(parser.values, option.dest, [value])

op = OptionParser(version=tool_version)

pg_option_group = OptionGroup(op,"Postgres","Postgres arguments to connect:")
pg_option_group.add_option("--pg_host",
              dest="pg_host", default = "localhost",
              help="Query Analysis Service Postgres database on HOST.", metavar="HOST")
pg_option_group.add_option("--pg_port",
              dest="pg_port", default = "2285", #"2282"
              help="Query Analysis Service Postgres database on PORT.", metavar="PORT")
pg_option_group.add_option("--pg_user",
              dest="pg_user", default = "operator",
              help="Query Analysis Service Postgres database with USER.", metavar="USER")
pg_option_group.add_option("--pg_password",
              dest="pg_password", default = "CastAIP",
              help="Query Analysis Service Postgres database with PASSWORD.", metavar="PASSWORD")
pg_option_group.add_option("--trace_pg",
              action="store_true", dest="trace_pg", default=False,
              help="Trace the queries sent to Postgres database.")

analysis_extraction_option_group = OptionGroup(op,"Analysis Service extraction","Analysis Service schema to perform extraction:")              
analysis_extraction_option_group.add_option("-s", "--schema_list",
              dest="schema_list", callback=arg_list_callback, type="string", action="callback",
              help="Use SCHEMA_LIST to query the Analysis Service(s) to process.", metavar="SCHEMA_LIST")


neo4j_option_group = OptionGroup(op,"Neo4j","Neo4j arguments to connect and feed back:")
neo4j_default_url = "bolt://localhost:7687"
#neo4j_location = r"C:\ProgramData\CAST1\ImagingSystem\Neo4j_data"
neo4j_location = "/home/smm/PhD/RoleExtraction/csv"
neo4j_option_group.add_option("--neo4j_import_location",
              dest="neo4j_import_location", default = neo4j_location,
              help="Neo4j import location.", metavar="URI")
neo4j_option_group.add_option("--neo4j_url",
              dest="neo4j_url", default = neo4j_default_url,
              help="Query and update Imaging System Neo4j database at URI.", metavar="URI")
neo4j_option_group.add_option("--neo4j_user",
              dest="neo4j_user", default = "neo4j",
              help="USER User to query and update Imaging System Neo4j database.", metavar="USER")
neo4j_option_group.add_option("--neo4j_password",
              dest="neo4j_password", default = "imaging",
              help="PASSWORD password for selected user to query and update Imaging System Neo4j database.", metavar="PASSWORD")
neo4j_option_group.add_option("--neo4j_database",
              dest="neo4j_database", default = "neo4j",
              help="DATABASE database to query and update Imaging System Neo4j database.", metavar="DATABASE")
neo4j_option_group.add_option("--trace_neo4j",
              action="store_true", dest="trace_neo4j", default=False,
              help="Trace the queries sent to Neo4j database.")

imaging_processing_option_group = OptionGroup(op,"Imaging content processing","Imaging arguments to control the process:")
imaging_processing_option_group.add_option("-a", "--application_list",
              dest="application_list", callback=arg_list_callback, type="string", action="callback",
              help="Limit processing to selected APPLICATION_LIST in Imaging System (omit to process them all).", metavar="APPLICATION_LIST")
imaging_processing_option_group.add_option("--min_community_size",
              dest="min_community_size", default = 10, type= "int",
              help="Filter out communities with less than MIN_COMMUNITY_SIZE elements.", metavar="MIN_COMMUNITY_SIZE")
imaging_processing_option_group.add_option("--source_list",
              dest="source_list", callback=arg_list_callback, type="string", action="callback",
              help="Feed on SOURCE_LIST list of text sources (among name,comment,mangling; omit to process them all).", metavar="SOURCE_LIST")

nlp_option_group = OptionGroup(op,"NLP tooling","NLP tooling arguments to connect and extract lemmas and part-of-speech information: ")
nlp_option_group.add_option("--min_word_size",
              dest="min_word_size", default = 2, type= "int",
              help="Filter out words with less than MIN_WORD_SIZE characters.", metavar="MIN_WORD_SIZE")
nlp_option_group.add_option("--pos_list",
              dest="pos_list", callback=arg_list_callback, type="string", action="callback",
              help="Keep words with POS in POS_LIST list of POS to keep (mostly among VERB,NOUN,ADJ,PROPN,ADV,NUM; omit to process VERB,NOUN,ADJ,PROPN).", metavar="POS_LIST")
nlp_option_group.add_option("--nlp_language",
              dest="nlp_language", default = "en",
              help="Use LANGUAGE when extracting lemma (default is 'en', also supported: 'fr', 'de', 'it', 'es')", metavar="LANGUAGE")

misc_option_group = OptionGroup(op,"Misc.","Misc. arguments:")
misc_option_group.add_option("--trace_panda",
              action="store_true", dest="trace_panda", default=False,
              help="Trace python panda dataframe handling.")

op.add_option_group(pg_option_group)
op.add_option_group(analysis_extraction_option_group)

op.add_option_group(neo4j_option_group)
op.add_option_group(imaging_processing_option_group)
op.add_option_group(nlp_option_group)
op.add_option_group(misc_option_group)

(options, args) = op.parse_args()


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')

def is_bundled():
    return getattr(sys, 'frozen', False)

# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)


print(__doc__)
op.print_version()
op.print_help()
print()
pp.pprint(vars(opts))
print()

# nlp check,  setting, and tool
if opts.nlp_language in ['fr','it','de','es']:
    spacy_model = "%s_core_news_sm-3.6.0" % (opts.nlp_language)
else:
    spacy_model = "en_core_web_sm"

try:
    logging.info(f"{is_bundled()} { __file__} here ")
    print(is_bundled(), __file__, "here")
    if is_bundled() and __file__:
        
        bundle_dir = Path(__file__).parent.absolute()
        logging.info('Running in bundle mode [%s]'%(bundle_dir))
        print(os.listdir(os.path.abspath(os.path.join(bundle_dir,spacy_model,spacy_model+'-3.6.0'))))
        nlp = spacy.load(os.path.abspath(os.path.join(bundle_dir,spacy_model,spacy_model+'-3.6.0')))
    else:
        import en_core_web_sm
        import fr_core_news_sm
        import de_core_news_sm
        import it_core_news_sm
        import es_core_news_sm
        if opts.nlp_language == 'fr':
            nlp = fr_core_news_sm.load()
        elif opts.nlp_language == 'de':
            nlp = de_core_news_sm.load()
        elif opts.nlp_language == 'it':
            nlp = it_core_news_sm.load()
        elif opts.nlp_language == 'es':
            nlp = es_core_news_sm.load()
        else:
            nlp = en_core_web_sm.load()

except:
    logging.info('')
    logging.critical('/!\\')
    logging.critical('Cannot load [%s] spacy model. Try "python -m spacy download %s" to install it then relaunch the tool.' % (spacy_model,spacy_model))
    logging.critical('/!\\')
    logging.info('')
    sys.exit()

# arguments check and process
if opts.pos_list is None:
    opts.pos_list = ['VERB','NOUN','ADJ','PROPN']
if opts.source_list is None:
    opts.source_list = ['name','mangling','comment']


# text processing
def clean_code(code, language="java"):
    """
    if not language:
        # Automatically detect the language
        try:
            lexer = pygments.lexers.guess_lexer(code)
            language = lexer.name
        except ClassNotFound:
            raise ValueError("Could not automatically detect language.")
    """

    lexer = lexers.get_lexer_by_name(language)
    tokens = lexer.get_tokens(code)
    
    cleaned_code = []
    for tok_type, tok_value in tokens:
        # Filter tokens based on type
        if tok_type in token.Comment:
            cleaned_code.append(tok_value)
        elif tok_type in token.Name and tok_type != token.Name.Builtin:
            cleaned_code.append(tok_value)
        elif tok_type in token.Literal.String:
            cleaned_code.append(tok_value)
        elif tok_type in token.Text and tok_value.strip():
            cleaned_code.append(tok_value)
    
    return ' '.join(cleaned_code)


def pre_processing(text):
    text = regex.sub('\\W',r' ', text)  # Remplace les caractères non-alphanumériques par des espaces
    text = regex.sub('[^\p{Latin} ]', r' ', text)  # Conserve seulement les lettres latines et les espaces
    # text = ''.join(c for c in text if c.isalnum() or c in '. ')
    text = regex.sub('((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))', r' \1', text).lower()  # Séparer les mots composés
    text = regex.sub('(?<=\\s)(\\w{1,2}\\s)', r' ', text)  # Supprimer les mots courts
    text = regex.sub('(?<=\\s)(\\d+\\s)', r' ', text)   # Supprimer les nombres
    text = regex.sub('\\b(\\w+)(?:\\W+\\1\\b)+', r' \1', text)   # Supprimer les répétitions
    text = regex.sub('\\s+', r' ', text)   # Remplacer les espaces multiples par un seul espace
    text = text.strip()  # Supprimer les espaces en début et fin de texte
    text = regex.sub('_', r' ', text)   # Remplacer les underscores par des espaces
    return text

"""
# List of programming keywords
programming_keywords = {
    'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', 
    'const', 'continue', 'default', 'do', 'double', 'else', 'enum', 'extends', 'final', 
    'finally', 'float', 'for', 'goto', 'if', 'implements', 'import', 'instanceof', 'int', 
    'interface', 'long', 'native', 'new', 'null', 'package', 'private', 'protected', 
    'public', 'return', 'short', 'static', 'strictfp', 'super', 'switch', 'synchronized', 
    'this', 'throw', 'throws', 'transient', 'try', 'void', 'volatile', 'while',
    'self', 'lambda', 'yield', 'async', 'await', 'namespace', 'require', 'include', 'define', 
    'export', 'global', 'instance', 'method', 'property', 'public', 'protected', 'private', 
    'constructor', 'defer', 'declaration', 'function', 'module', 'typeof', 'instanceof'
}

def filter_programming_keywords(text, keywords):
    #Remove programming keywords from the given text.
    # Use regex to remove the keywords, ensuring whole words are matched
    keyword_pattern = r'\b(?:' + '|'.join(map(re.escape, keywords)) + r')\b'
    filtered_text = re.sub(keyword_pattern, '', text)
    # Normalize whitespace
    filtered_text = re.sub(r'\s+', ' ', filtered_text).strip()
    return filtered_text
"""

def lemma_replacement(text):
    lemmatized_text = ''
    lemmatized_text = ' '.join([token.lemma_ for token in nlp(text)])
    return lemmatized_text

def post_processing(text):
    text = regex.sub('\\b(\\w+)(?:\\W+\\1\\b)+', r' \1', text)  # Supprimer les répétitions
    text = regex.sub('\\s+', r' ', text)  # Remplacer les espaces multiples par un seul espace
    text = text.strip()  # Supprimer les espaces en début et fin de texte
    return text

def lemma_pos_extraction(text):
    tokens = []
    response = nlp(text)
    unindexed_tokens = [dict(lemma=token.lemma_,pos=token.pos_) for token in response]
    for idx,val in enumerate(unindexed_tokens):
        val.update(dict(index=idx+1))
        tokens.append(val)
    return tokens


# Postgres acces
class PostgresConnection:
    def __init__(self, host, port, database, user, password):
        try:
            self.conn = psycopg2.connect(database=database, user=user, password = password, host = host, port = port)
        except Exception as e:
            print("Failed to connect:", e)

    def close(self):
        if self.conn is not None:
            self.conn.close()

    def query(self, query, trace = False):
        assert self.conn is not None, "Connection not initialized!"
        response = None
        if trace:
            print(query)
        try:
            response = sqlio.read_sql_query(query,self.conn)
        except Exception as e:
            print("Query failed:", e)

        return response


get_source_code_sql = """SELECT source_path, source_id, source_error, source_code, source_crc
	FROM :local_schema.dss_code_sources
"""

get_source_code_positions_sql = """SELECT object_id, source_id, line_start, line_end, col_start, col_end, panel
	FROM :local_schema.dss_source_positions
"""
get_source_code_to_be_cut_sql = """
SELECT 
	sp.object_id,
	sp.line_start, 
	sp.line_end,
    cs.source_code
FROM 
    :local_schema.dss_code_sources AS cs
JOIN 
    :local_schema.dss_source_positions AS sp
ON 
    cs.source_id = sp.source_id
"""

# Neo4j access
class Neo4jConnection:

    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd), encrypted=False)
        except Exception as e:
            logging.info('')
            logging.critical("/!\\")
            logging.critical("Failed to create the driver:", e)
            logging.critical("Check that you have authorized incoming connection over bolt protocol in $NEO4J_HOME/conf/neo4j.conf file with 'dbms.connector.bolt.enabled=true' and 'dbms.connector.bolt.listen_address=0.0.0.0:7687' (Neo4j service restart is required afterwards).")
            logging.critical("/!\\")
            logging.info('')
            sys.exit()

    def close(self):
        if self.__driver is not None:
            self.__driver.close()

    def query(self, query, db=None, params=None, trace=False):
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        if trace:
            logging.info(query)
            if params is not None:
                pp.pprint(params)
                logging.info(params)
        try:
            session = self.__driver.session(database=db) if db is not None else self.__driver.session()
            response = list(session.run(query=query,parameters=params)) if params is not None else list(session.run(query))
        except Exception as e:
            logging.info("Query failed: %s", e)
        finally:
            if session is not None:
                session.close()
        return response

# CYPHER queries
application_query_string = """MATCH (n:Application)<-[:HAS_APP]-(d:Domain {DBName:$dbname})
 RETURN n.Name AS ApplicationLabel """

remove_node_w_label_in_app_query_string = """
match (n)
where $node_label in labels(n)
and $target_app in labels(n)
CALL { WITH n
  DETACH DELETE n
} IN TRANSACTIONS OF 1000 ROWS;
"""


get_text_query_string = """
MATCH (op:ObjectProperty)-[p:Property]-(o) 
where $target_label in labels(o)
and (o:Object or o:SubObject or o:Column or o:Constraint)
and tolower(op.Description) contains 'comment' and not tolower(op.Description) contains 'number'
RETURN distinct o.Name as name, p.value as text, o.AipId as id, 'comment' as source
union all
MATCH (o) 
where $target_label in labels(o) 
and (o:Object or o:SubObject or o:Column or o:Constraint)
return distinct o.Name as name, o.Name as text, o.AipId as id, 'name' as source
union all
MATCH (o) 
where $target_label in labels(o) 
and (o:Object or o:SubObject) and o.Mangling is not null
return distinct o.Name as name, o.Mangling as text, o.AipId as id, 'mangling' as source
"""

get_text_query_string_cyclomatic_compexity_0 = """
MATCH (op:ObjectProperty)-[p:Property]-(o) 
WHERE $target_label in labels(o)
AND (o:Object OR o:SubObject OR o:Column OR o:Constraint)
AND tolower(op.Description) CONTAINS 'comment'
AND NOT tolower(op.Description) CONTAINS 'number'
AND EXISTS {
  MATCH (c:ObjectProperty)-[r:Property]-(o)
  WHERE c.Description = 'Cyclomatic Complexity'
  AND r.value <> '1'
}
RETURN DISTINCT o.Name AS name, p.value AS text, o.AipId AS id, 'comment' AS source
union all
MATCH (c:ObjectProperty)-[r:Property]-(o) 
where $target_label in labels(o) 
and (o:Object or o:SubObject or o:Column or o:Constraint)
and c.Description = "Cyclomatic Complexity"
and r.value <> '1'
return distinct o.Name as name, o.Name as text, o.AipId as id, 'name' as source
union all
MATCH (c:ObjectProperty)-[r:Property]-(o) 
where $target_label in labels(o) 
and (o:Object or o:SubObject or o:Column or o:Constraint)
and c.Description = "Cyclomatic Complexity"
and r.value <> '1'
return distinct o.Name as name, o.Mangling as text, o.AipId as id, 'mangling' as source
"""

get_text_query_string_cyclomatic_compexity_1 = """
MATCH (op:ObjectProperty)-[p:Property]-(o) 
WHERE $target_label in labels(o)
AND (o:Object OR o:SubObject OR o:Column OR o:Constraint)
AND tolower(op.Description) CONTAINS 'comment'
AND NOT tolower(op.Description) CONTAINS 'number'
AND NOT EXISTS {
  MATCH (c:ObjectProperty)-[r:Property]-(o)
  WHERE c.Description = 'Cyclomatic Complexity'
  AND r.value = '1'
}
RETURN DISTINCT o.Name AS name, p.value AS text, o.AipId AS id, 'comment' AS source
UNION ALL
MATCH (c:ObjectProperty)-[r:Property]-(o) 
WHERE $target_label in labels(o) 
AND (o:Object OR o:SubObject OR o:Column OR o:Constraint)
AND NOT EXISTS {
  MATCH (c:ObjectProperty)-[r:Property]-(o)
  WHERE c.Description = 'Cyclomatic Complexity'
  AND r.value = '1'
}
RETURN DISTINCT o.Name AS name, o.Name AS text, o.AipId AS id, 'name' AS source
UNION ALL
MATCH (c:ObjectProperty)-[r:Property]-(o) 
WHERE $target_label in labels(o) 
AND (o:Object OR o:SubObject OR o:Column OR o:Constraint)
AND NOT EXISTS {
  MATCH (c:ObjectProperty)-[r:Property]-(o)
  WHERE c.Description = 'Cyclomatic Complexity'
  AND r.value = '1'
}
RETURN DISTINCT o.Name AS name, o.Mangling AS text, o.AipId AS id, 'mangling' AS source
"""

get_neighborhood ="""
MATCH (c:ObjectProperty)-[r:Property]-(filteredNode)
WHERE $target_label IN labels(filteredNode)
AND (filteredNode:Object OR filteredNode:SubObject)
AND c.Description = "Cyclomatic Complexity"
AND r.value <> '1'
WITH collect(id(filteredNode)) AS nodeIds

MATCH (d:DataGraph)<-[:IS_IN_DATAGRAPH]-(n)
WHERE $target_label IN labels(d)
AND $target_label IN labels(n)
AND n.AipId = $target_aip
AND id(n) IN nodeIds
WITH DISTINCT d AS dataGraphs, nodeIds

MATCH (dataGraphs)<-[:IS_IN_DATAGRAPH]-(m)
WHERE $target_label IN labels(m)
AND (m:Object OR m:SubObject)
AND id(m) IN nodeIds
RETURN DISTINCT m.AipId
"""

update_neo4j_0 = """
MATCH (a {AipId: $node1}), (b {AipId: $node2})
WHERE $target_label IN labels(a)
AND $target_label IN labels(b)
CALL apoc.create.relationship(a, $relationship_type, $properties, b) YIELD rel
RETURN a, b, rel
"""

update_neo4j = """
MERGE (lc:LinkCategory {Name: "SEMANTIC"})

WITH lc
MATCH (a {AipId: $node1}), (b {AipId: $node2})
WHERE $target_label IN labels(a)
AND $target_label IN labels(b)
CALL apoc.create.relationship(a, $relationship_type, $properties, b) YIELD rel as r

WITH lc, a, b, r
MERGE (l:LinkType {Name: r.Name})
MERGE (l)<-[:Contains]-(lc)

RETURN a, b, r, l
"""

# init
print("""

╔╗           ╔═══╗╔═╗╔═╗╔═╗╔═╗             ╔╗ ╔╗                          ╔╗                    ╔═╗               ╔═══╗╔═══╗╔═══╗
║║           ║╔═╗║║║╚╝║║║║╚╝║║            ╔╝╚╗║║                          ║║                    ║╔╝               ║╔═╗║╚╗╔╗║║╔═╗║
║╚═╗╔╗ ╔╗    ║╚══╗║╔╗╔╗║║╔╗╔╗║    ╔╗╔╗╔╗╔╗╚╗╔╝║╚═╗    ╔══╗╔══╗╔╗╔╗╔══╗    ║║ ╔╗╔═╗ ╔══╗╔══╗    ╔╝╚╗╔═╗╔══╗╔╗╔╗    ║╚═╝║ ║║║║║║ ║║
║╔╗║║║ ║║    ╚══╗║║║║║║║║║║║║║    ║╚╝╚╝║╠╣ ║║ ║╔╗║    ║══╣║╔╗║║╚╝║║╔╗║    ║║ ╠╣║╔╗╗║╔╗║║══╣    ╚╗╔╝║╔╝║╔╗║║╚╝║    ║╔══╝ ║║║║║║ ║║
║╚╝║║╚═╝║    ║╚═╝║║║║║║║║║║║║║    ╚╗╔╗╔╝║║ ║╚╗║║║║    ╠══║║╚╝║║║║║║║═╣    ║╚╗║║║║║║║║═╣╠══║     ║║ ║║ ║╚╝║║║║║    ║║   ╔╝╚╝║║╚═╝║
╚══╝╚═╗╔╝    ╚═══╝╚╝╚╝╚╝╚╝╚╝╚╝     ╚╝╚╝ ╚╝ ╚═╝╚╝╚╝    ╚══╝╚══╝╚╩╩╝╚══╝    ╚═╝╚╝╚╝╚╝╚══╝╚══╝     ╚╝ ╚╝ ╚══╝╚╩╩╝    ╚╝   ╚═══╝╚═══╝
    ╔═╝║                                                                                                                         
    ╚══╝                                                                                                                         

                                                                                                                                            
""")

print("""

    ___  _____    
 .'/,-Y"     "~-.  
 l.Y             ^.           
 /\               _\_  
i            ___/"   "\ 
|          /"   "\   o !   
l         ]     o !__./   
 \ _  _    \.___./    "~\  
  X \/ \            ___./  
 ( \ ___.   _..--~~"   ~`-.  
  ` Z,--   /               \    
    \__.  (   /       ______) 
      \   l  /-----~~" /   
       Y   \          / 
       |    "x______.^ 
       |           \    
       |            \
                                                                                                                                            
""")

# connect

tot0 = time()
logging.info("Connection to Neo4j ...")
conn = Neo4jConnection(uri=opts.neo4j_url, user=opts.neo4j_user, pwd=opts.neo4j_password)
logging.info('')



logging.info("Querying application labels in [%s] database..."%(opts.neo4j_database))
t0 = time()
query_params = dict(dbname=opts.neo4j_database)
query_string = application_query_string
query_results = conn.query(query_string,params=query_params,db='imaging',trace=opts.trace_neo4j)
if query_results is None:
    logging.info('')
    logging.critical("/!\\")
    logging.critical("NO results")
    logging.critical("/!\\")
    logging.warning("[i]")
    logging.warning("Neo4j database name may differ from the Imaging tenant name as punctuation is not allowed")
    logging.warning("[i]")
    logging.info('')
    sys.exit()
elif len(query_results) == 0:
    logging.info('')
    logging.critical("/!\\")
    logging.critical("NO results")
    logging.critical("/!\\")
    logging.warning("[i]")
    logging.warning("Neo4j database name may differ from the Imaging tenant name as punctuation is not allowed")
    logging.warning("[i]")
    logging.info('')
    sys.exit()
else:
    application_label_dtf = pd.DataFrame([dict(_) for _ in query_results])
    logging.info('')
    logging.info("Application labels found:")
    logging.info(" ".join(application_label_dtf["ApplicationLabel"]))
    logging.info("Filtering on application(s) from [%s] application list..." % opts.application_list)
    logging.info('')
logging.info("done in %0.3fs." % (time() - t0))
logging.info('')

if opts.trace_panda:
    logging.info(application_label_dtf.info())


logging.info("Querying import path for [%s] database..."%(opts.neo4j_database))
t0 = time()

if not opts.neo4j_import_location or  not os.path.exists(opts.neo4j_import_location):
    logging.error('')
    logging.error("Invalid import path to neo4j: %s" % (opts.neo4j_import_location))
    logging.error('')
    sys.exit()
else:
    neo4j_import_path = opts.neo4j_import_location
    logging.info('')
    logging.info("Import path found: %s" % (neo4j_import_path))
    logging.info('')
logging.info("done in %0.3fs." % (time() - t0))
logging.info('')

local_import_path = neo4j_import_path
if opts.neo4j_url != neo4j_default_url and not os.path.isdir(neo4j_import_path):
    logging.info('')
    logging.warning("[i]")
    logging.warning("Neo4j database seems to be hosted remotely ([%s]). You will need to copy CSV files from [%s] on local machine to [%s] on the remote host at some point of the process." % (opts.neo4j_url,local_import_path,neo4j_import_path))
    logging.warning("[i]")
    logging.info('')
elif not os.path.isdir(neo4j_import_path):
    logging.info('')
    logging.critical("/!\\")
    logging.critical("NO directory")
    logging.critical("/!\\")
    logging.warning("[i]")
    logging.warning("Neo4j database import path is mandatory to complete the task: are you running the task on the Neo4j host machine?")
    logging.warning("[i]")
    logging.info('')
    sys.exit()


if opts.application_list is not None:
    logging.info("Filtering on application(s) from [%s] application list..." % opts.application_list)
    is_selected = application_label_dtf['ApplicationLabel'].isin(opts.application_list)
    if any(is_selected):
        application_label_dtf = application_label_dtf[is_selected]
    else:
        logging.warning("No application within [%s] application list found!" % opts.application_list)
        logging.warning("[i]")
        logging.warning("The name is case-sensitive.")
        logging.warning("[i]")
        logging.warning("=> processing all applications.")
    logging.info('')


for app in application_label_dtf["ApplicationLabel"]:
    logging.info("Clean-up sentences for [%s] database..."%(app))
    t0 = time()
    query_params = dict(target_app = app, node_label = 'Sentence')
    query_string = remove_node_w_label_in_app_query_string
    conn.query(query_string,params=query_params,db=opts.neo4j_database,trace=opts.trace_neo4j)
    logging.info("done in %0.3fs." % (time() - t0))
    logging.info('')

    logging.info("Clean-up tokens for [%s] database..."%(app))
    t0 = time()
    query_params = dict(target_app = app, node_label = 'Token')
    query_string = remove_node_w_label_in_app_query_string
    conn.query(query_string,params=query_params,db=opts.neo4j_database,trace=opts.trace_neo4j)
    logging.info("done in %0.3fs." % (time() - t0))
    logging.info('')

stop_word_files = []
for folder, subfolders, files in os.walk(os.getcwd()):
    for file in [f for f in files if re.match(r'.*\_keywords.txt', f) or f == 'stop_words.txt']:
        logging.info('Loading [%s] stop words, technical keywords file for ALL applications...' %(file))
        filePath = os.path.abspath(os.path.join(folder, file))
        stop_word_files.append(filePath)
all_stop_words = []
for f in stop_word_files:
    with open(f,'r') as fd:
        all_stop_words += [line.rstrip().lower() for line in fd.readlines()]
all_stop_words = list(set(all_stop_words))

for app in application_label_dtf["ApplicationLabel"]:
    translation_file = "%s_translations.csv" % app
    translation_dict = {}
    for folder, subfolders, files in os.walk(os.getcwd()):
        if translation_file in files:
            transaction_file_fn = os.path.abspath(os.path.join(folder, translation_file))
            logging.info('Loading [%s] translation file for [%s] application...' %(transaction_file_fn,app))
            with open(transaction_file_fn,'r') as fd:
                reader = csv.reader(fd)
                next(reader, None)  # skip the headers
                translation_dict = {rows[0]:rows[1] for rows in reader}

    def translate_text(text):
        translated_words=[]
        for word in text.split(' '):
            if word in translation_dict.keys():
                translated_words.append(translation_dict[word])
            else:
                translated_words.append(word)
        return ' '.join(translated_words)

    stop_words = []
    stop_words = [sw for sw in all_stop_words if sw not in translation_dict.keys()]

    logging.info('')
    logging.info("Querying text elements from [%s] application..."%(app))
    t0 = time()
    query_params = dict(target_label=app)
    query_string = get_text_query_string_cyclomatic_compexity_0
    query_results = conn.query(query_string,params=query_params,db=opts.neo4j_database,trace=opts.trace_neo4j)
    logging.info("done in %0.3fs." % (time() - t0))
    logging.info('')

    if query_results is None:
        logging.info('')
        logging.critical("/!\\")
        logging.critical("NO results")
        logging.critical("/!\\")
        logging.info('')
        sys.exit()
    elif len(query_results) == 0:
        logging.info('')
        logging.critical("/!\\")
        logging.critical("NO results")
        logging.critical("/!\\")
        logging.info('')
        sys.exit()
    else:
        application_text_dtf = pd.DataFrame([dict(_) for _ in query_results])
        if opts.trace_panda:
            logging.info(application_text_dtf.info())
        application_text_dtf = application_text_dtf[application_text_dtf['source'].isin(opts.source_list)]
        if opts.trace_panda:
            logging.info(application_text_dtf.info())


    def process_text(source_code, line_start, line_end):
        # Split the source code into lines
        lines = source_code.split('\n')
        # Extract the desired lines
        extracted_lines = lines[line_start-1:line_end]
        # Join the extracted lines back into a single string
        processed_text = '\n'.join(extracted_lines)
        return processed_text



    # Connecting to Postgres
    #for schema in opts['schema_list']:
    for schema in opts.schema_list:
        print("Connection to Postgres %s %s as %s ..." % (opts.pg_host,opts.pg_port,opts.pg_user))
        pgconn = PostgresConnection(host=opts.pg_host, 
                                    port = opts.pg_port, 
                                    user=opts.pg_user, 
                                    password=opts.pg_password, 
                                    database= "postgres" )  
        print()


        
        get_source_code_dtf = pgconn.query(get_source_code_to_be_cut_sql.replace(":local_schema", schema), trace=opts.trace_pg)
        #print(get_source_code_dtf.head(2))

        processed_data = []
        for idx, row in get_source_code_dtf.iterrows():
            object_id = row['object_id']
            line_start = row['line_start']
            line_end = row['line_end']
            source_code = row['source_code']
            
            processed_text = process_text(source_code, line_start, line_end)
            #print(f"processed_text: {processed_text}")


            # Append the processed row to the list
            processed_data.append({'object_id': object_id, 'source_code_cut': processed_text})
            #pgconn.execute(store_processed_data_sql.replace(":object_id", str(object_id)).replace(":processed_text", processed_text), trace=opts['trace_pg'])

        # Create a DataFrame from the list of processed data
        store_data_dtf = pd.DataFrame(processed_data)

        if opts.trace_panda:
            print(get_source_code_dtf.info())

        print("Processed and stored data for schema:", schema)
        print()

    #print(application_text_dtf.head(2))
    #print(store_data_dtf.head(2))

    # Convert 'object_id' in store_data_dtf to string
    store_data_dtf['object_id'] = store_data_dtf['object_id'].astype(str)

    # Create a mapping from `id` to `name` using the existing `application_text_dtf`
    id_to_name = application_text_dtf.set_index('id')['name'].to_dict()

    # Filter store_data_dtf to only include rows where object_id is in application_text_dtf id
    filtered_store_data = store_data_dtf[store_data_dtf['object_id'].isin(application_text_dtf['id'])]

    # Prepare the rows to be added to application_text_dtf
    new_rows = []
    for idx, row in filtered_store_data.iterrows():
        object_id = row['object_id']
        source_code_cut = row['source_code_cut']
        
        # Retrieve the `name` using `object_id`
        name = id_to_name.get(object_id, 'Unknown')
        
        new_row = {
            'name': name,
            'text': source_code_cut,
            'id': object_id,
            'source': 'source_code'
        }
        new_rows.append(new_row)

    # Create a DataFrame for the new rows
    new_rows_df = pd.DataFrame(new_rows)

    # Use pd.concat to append the new rows to application_text_dtf
    application_text_dtf = pd.concat([application_text_dtf, new_rows_df], ignore_index=True)

    # Print the updated DataFrame
    #print(application_text_dtf.head(2))



    logging.info("Processing text elements from [%s] application..." % (app))
    ot0 = time()

    t0 = time()
    application_text_dtf['cleanedText'] = application_text_dtf["text"].map(clean_code)
    logging.info("... clean done in %0.3fs." % (time() - t0))
    if opts.trace_panda:
        logging.info(application_text_dtf.info())

    t0 = time()
    application_text_dtf['processedText'] = application_text_dtf["cleanedText"].map(pre_processing)
    logging.info("... split done in %0.3fs." % (time() - t0))
    if opts.trace_panda:
        logging.info(application_text_dtf.info())
    
    t0 = time()
    application_text_dtf['filteredText'] = application_text_dtf['processedText'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    logging.info("... filtering done in %0.3fs." % (time() - t0))
    if opts.trace_panda:
        logging.info(application_text_dtf.info())
    
    t0 = time()
    application_text_dtf['translatedText'] = application_text_dtf['filteredText'].map(translate_text)
    logging.info("... translation done in %0.3fs." % (time() - t0))
    if opts.trace_panda:
        logging.info(application_text_dtf.info())
    
    t0 = time()
    application_text_dtf['replacedText'] = application_text_dtf["translatedText"].map(lemma_replacement)
    logging.info("... lemmatization done in %0.3fs." % (time() - t0))
    if opts.trace_panda:
        logging.info(application_text_dtf.info())
    
    t0 = time()
    application_text_dtf['sentence'] = application_text_dtf["replacedText"].map(post_processing)
    logging.info("... finalization done in %0.3fs." % (time() - t0))
    if opts.trace_panda:
        logging.info(application_text_dtf.info())
    
    """
    t0 = time()
    application_text_dtf['annotatedText'] = application_text_dtf["sentence"].map(lemma_pos_extraction)
    logging.info("... POS done in %0.3fs." % (time() - t0))
    if opts.trace_panda:
        logging.info(application_text_dtf.info())
    """

    logging.info("done in %0.3fs." % (time() - ot0))
    logging.info('')

    
    
    """
    # Print the first two rows of the filtered DataFrame
    filtered_df = application_text_dtf[application_text_dtf['source'] == 'source_code']
    print(filtered_df.head(3))

    # Save the string to a .txt file
    first_3_rows = filtered_df.head(3)
    output_string = first_3_rows.to_string(index=False)
    with open('filtered_rows.txt', 'w') as file:
        file.write(output_string)

    # Print all available columns and their data types
    print(application_text_dtf.dtypes)
    """
    """
    def extract_keywordsold(text):
    r.extract_keywords_from_text(text)
    keywords = r.get_ranked_phrases_with_scores()
    print(f"keywords : {keywords}")
    best_keywords = keywords[0][1]
    return best_keywords
    """

    t0 = time()
    from rake_nltk import Rake

    pd.set_option('display.max_columns', None)

    # Function to merge sentences and names by id
    def merge_sentences_with_name(df):
        merged_df = df.groupby('id').agg({
            'name': 'first',
            'sentence': lambda x: f"{df.loc[x.index, 'name'].iloc[0]}\n" + '\n'.join(x)
        }).reset_index()
        return merged_df
    
    # Function to merge sentences by id
    def merge_sentences(df):
        merged_df = df.groupby('id').agg({
            'name': 'first',
            'sentence': ' '.join
        }).reset_index()
        return merged_df

    # Apply the merging function
    merged_df = merge_sentences(application_text_dtf)

    # Function to extract keywords (provided)
    def extract_keywords(text):
        #print(f"{text}\n")
        r = Rake()
        r.extract_keywords_from_text(text)
        keywords_with_scores = r.get_ranked_phrases_with_scores()
        if not keywords_with_scores:
            return []
        #highest_score = max(score for score, _ in keywords_with_scores)
        #best_keywords = [keyword for score, keyword in keywords_with_scores if score == highest_score]
        #return best_keywords
        return keywords_with_scores[0][1]

    from sklearn.feature_extraction.text import TfidfVectorizer

    def extract_keywords_tfidf(text, num_keywords=5):
        """
        Extracts keywords from a given text using TF-IDF.

        Parameters:
        text (str): The text to extract keywords from.
        num_keywords (int): The number of top keywords to return. Default is 5.

        Returns:
        list: A list of top keywords.
        """
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=num_keywords)
        tfidf_matrix = tfidf_vectorizer.fit_transform([text])
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        # Get the scores for the features
        tfidf_scores = tfidf_matrix.toarray().flatten()
        
        # Sort the feature names by their scores in descending order
        top_keywords = [feature_names[i] for i in tfidf_scores.argsort()[-num_keywords:][::-1]]
        
        return top_keywords

    # Apply keyword extraction to the merged sentences
    #merged_df['keywords'] = merged_df['sentence'].apply(extract_keywords)
    merged_df['keywords'] = merged_df['sentence'].apply(extract_keywords_tfidf)


    # Final DataFrame
    Role_df = merged_df[['id', 'name', 'sentence', 'keywords']]
    
    logging.info("... keywords extraction done in %0.3fs." % (time() - t0))
    if opts.trace_panda:
        logging.info(application_text_dtf.info())
    

    keywords_file_name = '%s_keywords4.csv' % (app)
    if opts.neo4j_url != neo4j_default_url:
        keywords_file_name = os.path.abspath(os.path.join(local_import_path,keywords_file_name))
    else:
        keywords_file_name = os.path.abspath(os.path.join(neo4j_import_path,keywords_file_name))
    logging.info("Saving keywords df to [%s] file for [%s] application..." % (keywords_file_name,app))
    t0 = time()
    #Role_df[Role_df['keywords']!=''].drop_duplicates().to_csv(keywords_file_name,sep=';',index=False)
    Role_df[Role_df['keywords']!=''].to_csv(keywords_file_name,sep=';',index=False)
    logging.info("done in %0.3fs." % (time() - t0))
    logging.info('')


    '''
    # Filter the DataFrame where 'id' is '379561'
    filtered_df = Role_df[Role_df['id'] == '379561']
    output_string = filtered_df.to_string(index=False)
    with open('role379561.txt', 'w') as file:
        file.write(output_string)
    '''

    t0 = time()


    from itertools import combinations
    
    # Create the graph
    G = nx.Graph()

    # Add nodes with 'id', 'name' and 'keywords' as attributes
    for _, row in Role_df.iterrows():
        G.add_node(row['id'], name=row['name'], keywords=row['keywords'])

    """
    # Function to calculate Jaccard similarity between two sets of keywords
    def jaccard_similarity(keywords1, keywords2):
        set1 = set(keywords1)
        set2 = set(keywords2)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        if not union:  # Handle the case where both sets are empty
            return 0
        return len(intersection) / len(union)

    # Add edges based on keyword similarity
    threshold = 0.25  # threshold for similarity to create an edge
   
    print("Before combi")
    for id1, id2 in combinations(Role_df['id'], 2):
        keywords1 = Role_df[Role_df['id'] == id1]['keywords'].values[0]
        keywords2 = Role_df[Role_df['id'] == id2]['keywords'].values[0]
        
        if keywords1 is None or keywords2 is None:
            continue
        
        similarity = jaccard_similarity(keywords1, keywords2)
        if similarity > threshold:
            G.add_edge(id1, id2, weight=similarity)

    print("After combi")
    """

    def append_query_results_to_dict(id, query_results, D):
        # Extracting the m.AipId values from the query results
        aip_ids = [record["m.AipId"] for record in query_results]
        # Adding the id and aip_ids to the dictionary D
        D[id] = aip_ids
        return D


    logging.info('')
    logging.info("Querying neighborhoods from [%s] application..."%(app))
    t0 = time()

    D = {}
    for id in Role_df["id"]:
    #for id in Role_df["id"].head(2):
        #id = Role_df["id"].iloc[1]
        query_params = dict(target_label = app, target_aip = id)
        query_string = get_neighborhood
        #print(query_string)
        query_results = conn.query(query_string,params=query_params,db=opts.neo4j_database,trace=opts.trace_neo4j)
        #print(f"query_results : {query_results}")
        D = append_query_results_to_dict(id, query_results, D)

    logging.info("done in %0.3fs." % (time() - t0))
    logging.info('')

    from gensim.models import KeyedVectors
    from sklearn.metrics.pairwise import cosine_similarity

    # Load the model
    word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    # Function to get the aggregate vector for a node
    def get_node_vector(keywords, model):
        vectors = []
        for word in keywords:
            if word in model:
                vectors.append(model[word])
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(len(next(iter(model.values()))))  # Model vector size

    # Compute vector for each node
    node_vectors = {}
    for node in G.nodes:
        keywords = G.nodes[node]['keywords']
        node_vectors[node] = get_node_vector(keywords, word_vectors)

    #print(node_vectors)
    #print(type(node_vectors))
    #print(type(node_vectors['9887']))

    # Define a threshold for similarity
    similarity_threshold = 0.65
    """
    # Compute cosine similarity and add edges based on the threshold
    #for node1 in G.nodes:
    #    for node2 in G.nodes:
    for node1 in D:
        print(f"type(node1) : {type(node1)}")
        for node2 in D[node1]:
            print(f"type(node2) : {type(node2)}")
            if node1 < node2:  # Avoid duplicate checking
                sim = cosine_similarity([node_vectors[node1]], [node_vectors[node2]])[0][0]
                if sim >= similarity_threshold:
                    G.add_edge(node1, node2, weight=sim)

    # Output the edges of the graph with their similarity scores
    for edge in G.edges(data=True):
        print(edge)
    """

    #print(f"Keys in node_vectors: {list(node_vectors.keys())[:5]}")

    # Function to find common keywords between two nodes
    def find_common_keywords(keywords1, keywords2):
        return list(set(keywords1).intersection(set(keywords2)))
    
    # Function to create an edge name from common keywords
    def create_edge_name(common_keywords):
        return ''.join([word.capitalize() for word in common_keywords])

    # Compute cosine similarity and add edges based on the threshold
    for node1 in D:
        node1_str = str(node1)  # Convert node1 to string, ensure it matches the graph ID format
        if node1_str not in node_vectors:
            print(f"Node {node1_str} not found in node_vectors")
            continue

        for node2 in D[node1]:
            node2_str = str(node2)  # Convert node2 to string, ensure it matches the graph ID format
            if node2_str not in node_vectors:
                print(f"Node {node2_str} not found in node_vectors")
                continue

            if node1_str < node2_str:  # Avoid duplicate checking
                sim = cosine_similarity([node_vectors[node1_str]], [node_vectors[node2_str]])[0][0]
                if sim >= similarity_threshold:
                    #G.add_edge(node1_str, node2_str, weight=sim)
                    # Find common keywords
                    keywords1 = G.nodes[node1_str]['keywords']
                    keywords2 = G.nodes[node2_str]['keywords']
                    common_keywords = find_common_keywords(keywords1, keywords2)

                    # Create edge name from common keywords
                    edge_name = create_edge_name(common_keywords)
                    
                    # Define edge properties
                    edge_properties = {
                        'Weight': sim,
                        'Name': edge_name,
                        'CommonKeywords': common_keywords,
                    }
                    
                    # Add edge with properties
                    G.add_edge(node1_str, node2_str, **edge_properties)

        # Output the edges of the graph with their similarity scores
    for edge in G.edges(data=True):
        print(edge)

    # Print the graph information
    print(G)

    logging.info("Semantic Graph creation done in %0.3fs." % (time() - t0))
    logging.info('')

    logging.info("Updating Neo4j database with new edges for [%s] application..." % (app))
    t0 = time()
    
    for node1, node2, edge_data in G.edges(data=True):
        #relationship_type = edge_data.get('Name')
        relationship_type = "SEMANTIC_SIM"
        edge_properties = {key: value for key, value in edge_data.items() if key != 'type'}
        
        # Construct the Cypher query
        query_string = update_neo4j
        
        # Query parameters
        query_params = dict(
            target_label=app,
            node1=node1,
            node2=node2,
            relationship_type=relationship_type,
            properties=edge_properties
        )
        
        # Execute the query
        conn.query(query_string, params=query_params, db=opts.neo4j_database)
    
    logging.info("Update complete in %0.3fs." % (time() - t0))
    logging.info('')



logging.info("Disconnection from Neo4j ...")
conn.close()
logging.info('')

logging.info("Disconnection from Postgres ...")
pgconn.close()
logging.info('')

logging.info("total ellapsed time: %0.3fs." % (time() - tot0))
logging.info('')

print("""
██████╗  ██████╗ ███╗   ██╗███████╗
██╔══██╗██╔═══██╗████╗  ██║██╔════╝
██║  ██║██║   ██║██╔██╗ ██║█████╗  
██║  ██║██║   ██║██║╚██╗██║██╔══╝  
██████╔╝╚██████╔╝██║ ╚████║███████╗
╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚══════╝
   """)
