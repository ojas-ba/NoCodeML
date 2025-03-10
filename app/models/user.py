from pydantic import BaseModel,EmailStr,StrictInt,Field

class UserRegisteration(BaseModel):
    email: EmailStr
    firstName: str = Field(...,min_length=1,max_length=50)
    lastName: str = Field(...,min_length=1,max_length=50)
    password: str = Field(...,min_length=8,max_length=50)
    secretKey: StrictInt = Field(...,gt=999,lt=10000)

class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(...,min_length=8,max_length=50)
    