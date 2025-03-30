from fastapi import APIRouter,Depends, HTTPException,Response,Request
from ..models.user import UserRegisteration,UserLogin
import asyncpg
from ..db.init_db import get_db_connection
from .auth_funtions import hashing_password,verify_password,create_access_token,verify_token

router  = APIRouter()

@router.get("/me",status_code=200)
async def read_users_me(request:Request,db: asyncpg.Connection = Depends(get_db_connection)):
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    # Verify the token
    payload = verify_token(token)  # verify_token expects a string (token) it returns email
    
    sql_query = """
    SELECT id FROM Users
    WHERE email = $1;
    """
    try:
        result = await db.fetchval(sql_query,payload)
        if result:
            return {"user":payload,"user_id": result}
        else:
            return {"message": "User not found"}
    except Exception as e:
        return {"message": str(e)}
    

@router.post("/register",status_code=201)
async def register_user(user: UserRegisteration,db:asyncpg.Connection = Depends(get_db_connection)):
    
    sql_query = """
    INSERT INTO Users (email, first_name, last_name, password, secret_key)
    VALUES ($1::VARCHAR(255), $2::VARCHAR(50), $3::VARCHAR(50), $4::VARCHAR(1000), $5::INT)
    RETURNING id;
    """
    try:
       hash_password = hashing_password(user.password)
       user_id = await db.fetchval(
                                    sql_query,
                                    user.email,          # $1 -> email
                                    user.firstName,      # $2 -> first_name
                                    user.lastName,       # $3 -> last_name
                                    hash_password,       # $4 -> password
                                    user.secretKey       # $5 -> secret_key
                                    )
       if user_id:
        return {"message": "User created successfully"}
    except asyncpg.exceptions.UniqueViolationError:
        return {"message": "Email already exists"}
    except Exception as e:
        return {"message": str(e)}

@router.post("/login",status_code=200)
async def login_user(response:Response,user: UserLogin,db:asyncpg.Connection = Depends(get_db_connection)):
    sql_query = """
    SELECT password FROM Users
    WHERE email = $1;
    """
    
    try:
        original_hashed_password = await db.fetchval(sql_query,user.email)
        
        if original_hashed_password and verify_password( user.password , original_hashed_password ):
            access_token = create_access_token(user)
            response.set_cookie(key="access_token",value=access_token,httponly=True,samesite="lax",secure=False) # For production using secure True. It strictly nforces https
            return {"message":"success"}
            
        else:
            return {"message": "Invalid Credentials"}
        
    except Exception as e:
        return {"message": str(e)}
    
    
@router.post("/logout",status_code=200)
async def logout_user(response: Response):
    # Clear the access token cookie
    response.delete_cookie(key="access_token")
    
    return {"message": "Logout successful"}

    