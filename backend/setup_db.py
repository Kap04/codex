import os
from dotenv import load_dotenv
from supabase import create_client

# Load environment variables
load_dotenv()

# Supabase setup
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")

if not supabase_url or not supabase_key:
    print("Error: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env file")
    exit(1)

supabase = create_client(supabase_url, supabase_key)

def setup_database():
    """Create necessary tables if they don't exist"""
    try:
        print("Setting up database tables...")
        
        # Create users table using SQL
        print("Creating users table...")
        # Use the rpc method to execute SQL
        result = supabase.rpc(
            "create_users_table",
            {
                "sql": """
                CREATE TABLE IF NOT EXISTS public.users (
                    id UUID PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                """
            }
        ).execute()
        print(f"Result: {result}")
        
        # Alternative approach: Try to insert a test user and then delete it
        print("Testing table access by inserting and deleting a test user...")
        try:
            # Insert a test user
            test_user = supabase.table("users").insert({
                "id": "00000000-0000-0000-0000-000000000000",
                "email": "test@example.com",
                "password": "test_password"
            }).execute()
            print(f"Test user inserted: {test_user}")
            
            # Delete the test user
            delete_result = supabase.table("users").delete().eq("id", "00000000-0000-0000-0000-000000000000").execute()
            print(f"Test user deleted: {delete_result}")
            
            print("Users table exists and is accessible")
        except Exception as e:
            print(f"Error testing table: {str(e)}")
            
        print("Database setup completed")
    except Exception as e:
        print(f"Error setting up database: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    setup_database() 