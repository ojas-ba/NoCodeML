import jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext
from ..models.user import UserLogin

SECRET_KEY_JWT = "NOCODEML_AGILE"
ALGORITHM = "HS256"

pwd_context = CryptContext(schemes=["bcrypt"], default="bcrypt")# this is for hashing

def hashing_password(password:str):
    return pwd_context.hash(password)
def verify_password(password:str,hashed_password:str):
    return pwd_context.verify(password,hashed_password)

def create_access_token(user:UserLogin):
    data_to_encode = user.model_copy()
    expire = datetime.utcnow()+timedelta(hours=12)
    payload = {
        "sub":data_to_encode.email,
        "exp":expire
    }
    return jwt.encode(payload, SECRET_KEY_JWT, algorithm=ALGORITHM)

def verify_token(token:str):
    try:
        payload = jwt.decode(token,SECRET_KEY_JWT,algorithms=[ALGORITHM])
        email = payload["sub"]
        return email
    except jwt.ExpiredSignatureError:
        return "Expired Token"
    except jwt.InvalidTokenError:
        return "Invalid Token"
    
"""
Notes:
- The payload is a dictionary that contains the user's email and the expiration time of the token.
- The token is encoded using the HS256 algorithm and the SECRET_KEY_JWT as the secret key
- The token is returned as a string.
- sub means subject i can use payload.get("sub") to extract the email(if required)

"""