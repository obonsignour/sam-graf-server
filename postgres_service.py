import psycopg2
import psycopg2.extras
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class PostgresService:
    """Service for interacting with PostgreSQL database"""
    
    def __init__(self):
        """Initialize with connection parameters from environment variables"""
        self.host = os.getenv('PG_HOST', 'localhost')
        self.port = os.getenv('PG_PORT', '2285')
        self.user = os.getenv('PG_USER', 'operator')
        self.password = os.getenv('PG_PASSWORD', 'CastAIP')
        self.database = os.getenv('PG_DATABASE', 'postgres')
        self.schema = os.getenv('PG_SCHEMA', 'ecommerce_local')
        self.conn = None

    def connect(self):
        """Establish a connection to the PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database
            )
            return True
        except Exception as e:
            print(f"Error connecting to PostgreSQL: {e}")
            return False

    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None

    def get_source_code(self, object_id):
        """
        Get the source code for an object by its ID
        
        Args:
            object_id (str): The AipId of the object
            
        Returns:
            dict: Contains source code and metadata
        """
        if not self.conn and not self.connect():
            return {"error": "Database connection failed"}
        
        try:
            # First, get the source position for the object
            position_query = f"""
            SELECT object_id, source_id, line_start, line_end, col_start, col_end, panel
            FROM {self.schema}.dss_source_positions
            WHERE object_id = %s
            """
            
            cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) # type: ignore
            cursor.execute(position_query, (object_id,))
            position_data = cursor.fetchone()
            
            if not position_data:
                return {"sourceCode": None, "error": "No source position found for this object"}
            
            # Next, get the source code using the source_id
            code_query = f"""
            SELECT source_path, source_id, source_code
            FROM {self.schema}.dss_code_sources
            WHERE source_id = %s
            """
            
            cursor.execute(code_query, (position_data['source_id'],))
            code_data = cursor.fetchone()
            
            if not code_data:
                return {"sourceCode": None, "error": "No source code found for this object"}
            
            # Extract the relevant portion of the source code based on line numbers
            lines = code_data['source_code'].split('\n')
            line_start = max(0, position_data['line_start'] - 1)  # Adjust for 0-based indexing
            line_end = min(len(lines), position_data['line_end'])
            
            source_code_excerpt = '\n'.join(lines[line_start:line_end])
            
            return {
                "sourceCode": source_code_excerpt,
                "sourcePath": code_data['source_path'],
                "lineStart": position_data['line_start'],
                "lineEnd": position_data['line_end']
            }
            
        except Exception as e:
            print(f"Error fetching source code: {e}")
            return {"sourceCode": None, "error": str(e)}
        finally:
            cursor.close()