import psycopg2
import psycopg2.extras
from typing import Optional, Dict, Any
import logging
import uuid

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection: Optional[psycopg2.extensions.connection] = None
        
    def connect(self) -> bool:
        try:
            self.connection = psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['username'],
                password=self.config['password']
            )
            logger.info("PostgreSQL connection established successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            return False
    
    def disconnect(self):
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("PostgreSQL connection closed")
    
    def is_connected(self) -> bool:
        try:
            if self.connection and not self.connection.closed:
                cur = self.connection.cursor()
                cur.execute("SELECT 1")
                cur.fetchone()
                cur.close()
                return True
        except:
            pass
        return False
    
    def update_car_info(self, session_id: str, brand: str, model: str, body_type: str) -> bool:
        if not self.is_connected():
            if not self.connect():
                return False
        
        try:
            cur = self.connection.cursor()
            query = """
            INSERT INTO car_frontal_info (session_id, car_brand, car_model, car_body_type, updated_at)
            VALUES (%s, %s, %s, %s, NOW())
            ON CONFLICT (session_id) 
            DO UPDATE SET 
                car_brand = EXCLUDED.car_brand,
                car_model = EXCLUDED.car_model,
                car_body_type = EXCLUDED.car_body_type,
                updated_at = NOW()
            """
            cur.execute(query, (session_id, brand, model, body_type))
            self.connection.commit()
            cur.close()
            logger.info(f"Updated car info for session {session_id}: {brand} {model} ({body_type})")
            return True
        except Exception as e:
            logger.error(f"Failed to update car info: {e}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def execute_update(self, table: str, key_field: str, key_value: str, fields: Dict[str, str]) -> bool:
        if not self.is_connected():
            if not self.connect():
                return False
        
        try:
            cur = self.connection.cursor()
            
            # Build the UPDATE query dynamically
            set_clauses = []
            values = []
            
            for field, value in fields.items():
                if value == "NOW()":
                    set_clauses.append(f"{field} = NOW()")
                else:
                    set_clauses.append(f"{field} = %s")
                    values.append(value)
            
            # Add schema prefix if table doesn't already have it
            full_table_name = table if '.' in table else f"gas_station_1.{table}"
            
            query = f"""
            INSERT INTO {full_table_name} ({key_field}, {', '.join(fields.keys())})
            VALUES (%s, {', '.join(['%s'] * len(fields))})
            ON CONFLICT ({key_field})
            DO UPDATE SET {', '.join(set_clauses)}
            """
            
            # Add key_value to the beginning of values list
            all_values = [key_value] + list(fields.values()) + values
            
            cur.execute(query, all_values)
            self.connection.commit()
            cur.close()
            logger.info(f"Updated {table} for {key_field}={key_value}")
            return True
        except Exception as e:
            logger.error(f"Failed to execute update on {table}: {e}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def create_car_frontal_info_table(self) -> bool:
        """Create the car_frontal_info table in gas_station_1 schema if it doesn't exist."""
        if not self.is_connected():
            if not self.connect():
                return False
        
        try:
            cur = self.connection.cursor()
            
            # Create schema if it doesn't exist
            cur.execute("CREATE SCHEMA IF NOT EXISTS gas_station_1")
            
            # Create table if it doesn't exist
            create_table_query = """
            CREATE TABLE IF NOT EXISTS gas_station_1.car_frontal_info (
                display_id VARCHAR(255),
                captured_timestamp VARCHAR(255),
                session_id VARCHAR(255) PRIMARY KEY,
                license_character VARCHAR(255) DEFAULT NULL,
                license_type VARCHAR(255) DEFAULT 'No model available',
                car_brand VARCHAR(255) DEFAULT NULL,
                car_model VARCHAR(255) DEFAULT NULL,
                car_body_type VARCHAR(255) DEFAULT NULL,
                updated_at TIMESTAMP DEFAULT NOW()
            )
            """
            
            cur.execute(create_table_query)
            
            # Add columns if they don't exist (for existing tables)
            alter_queries = [
                "ALTER TABLE gas_station_1.car_frontal_info ADD COLUMN IF NOT EXISTS car_brand VARCHAR(255) DEFAULT NULL",
                "ALTER TABLE gas_station_1.car_frontal_info ADD COLUMN IF NOT EXISTS car_model VARCHAR(255) DEFAULT NULL", 
                "ALTER TABLE gas_station_1.car_frontal_info ADD COLUMN IF NOT EXISTS car_body_type VARCHAR(255) DEFAULT NULL",
                "ALTER TABLE gas_station_1.car_frontal_info ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW()"
            ]
            
            for alter_query in alter_queries:
                try:
                    cur.execute(alter_query)
                    logger.debug(f"Executed: {alter_query}")
                except Exception as e:
                    # Ignore errors if column already exists (for older PostgreSQL versions)
                    if "already exists" in str(e).lower():
                        logger.debug(f"Column already exists, skipping: {alter_query}")
                    else:
                        logger.warning(f"Error in ALTER TABLE: {e}")
            
            self.connection.commit()
            cur.close()
            logger.info("Successfully created/verified car_frontal_info table with all required columns")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create car_frontal_info table: {e}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def insert_initial_detection(self, display_id: str, captured_timestamp: str, session_id: str = None) -> str:
        """Insert initial detection record and return the session_id."""
        if not self.is_connected():
            if not self.connect():
                return None
        
        # Generate session_id if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        try:
            # Ensure table exists
            if not self.create_car_frontal_info_table():
                logger.error("Failed to create/verify table before insertion")
                return None
            
            cur = self.connection.cursor()
            insert_query = """
            INSERT INTO gas_station_1.car_frontal_info 
            (display_id, captured_timestamp, session_id, license_character, license_type, car_brand, car_model, car_body_type)
            VALUES (%s, %s, %s, NULL, 'No model available', NULL, NULL, NULL)
            ON CONFLICT (session_id) DO NOTHING
            """
            
            cur.execute(insert_query, (display_id, captured_timestamp, session_id))
            self.connection.commit()
            cur.close()
            logger.info(f"Inserted initial detection record with session_id: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to insert initial detection record: {e}")
            if self.connection:
                self.connection.rollback()
            return None