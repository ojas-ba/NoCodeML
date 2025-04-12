# This is used for automatic authentication of each route other than login and register. 
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import jwt

SECRET_KEY = "NOCODEML_AGILE"
ALGORITHM = "HS256"

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        
        if request.method == "OPTIONS":
            return await call_next(request)
        
        if request.url.path.startswith("/docs_swagger"):
            return await call_next(request)
        if request.url.path in ["/auth/login", "/auth/register", "/auth/me","/auth/logout","/","/openapi.json","/check-celery","/favicon.ico"]:
            return await call_next(request)

        token = request.cookies.get("access_token")
        if not token:
            raise HTTPException(status_code=401, detail="Missing token")
        try:
            jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

        return await call_next(request)


"""
You can add it like this in main.py. It works like a charm. 
# Initialize FastAPI with middleware
app = FastAPI()
app.add_middleware(AuthMiddleware)
"""